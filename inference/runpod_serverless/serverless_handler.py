#!/usr/bin/env python3
"""
RunPod Serverless Handler - Footstep Audio Generator

This handler loads the LoRA model once on cold start, then serves
generation requests via RunPod's serverless API.

Architecture:
- Cold start: Load model (10-15 seconds, happens once)
- Warm requests: Generate audio (20-30 seconds per request)
- Auto-scales: Container shuts down after ~60s idle (costs $0)
"""

import runpod
import torch
import json
import base64
import io
import sys
import traceback
import os
from pathlib import Path

# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================

from huggingface_hub import login

# Authenticate with Hugging Face using token from environment
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    try:
        login(token=hf_token)
        print("✓ Authenticated with Hugging Face")
    except Exception as e:
        print(f"⚠ Warning: HF authentication failed: {e}")
else:
    print("⚠ Warning: HF_TOKEN not found - model download may fail for gated models")

# ============================================================================
# IMPORTS
# ============================================================================

# Add paths for imports (only needed if using git-cloned repos)
sys.path.insert(0, "/app/LoRAW")

from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.sampling import sample_k, sample_rf
from loraw.network import create_lora_from_config
import torchaudio
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

CHECKPOINT_PATH = "/app/best.ckpt"
CONFIG_PATH = "/app/model_config_lora.json"
PRETRAINED_NAME = "stabilityai/stable-audio-open-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# GLOBAL MODEL (Loaded once on cold start)
# ============================================================================

print("=" * 80)
print("COLD START: Loading model...")
print("=" * 80)

# Global variables to be initialized
model = None
lora = None
sample_rate = None
sample_size = None
latent_sample_size = None

# Load base model
try:
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n[1/4] Loading base model from HuggingFace...")
    model, base_model_config = get_pretrained_model(PRETRAINED_NAME)
    model = model.to(DEVICE).eval().requires_grad_(False)
    print("✓ Base model loaded")
    
    print("\n[2/4] Loading training config...")
    with open(CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    
    # Merge configs
    model_config = base_model_config.copy()
    model_config['lora'] = training_config['lora']
    
    # lora model was trained on 6s (265,000 samples)
    model_config['sample_size'] = training_config.get('sample_size', 265000)
    model_config['sample_rate'] = training_config.get('sample_rate', 44100)
    
    print("✓ Config loaded")
    print(f"  Using sample_size: {model_config['sample_size']} (from training config)")
    
    print("\n[3/4] Loading LoRA checkpoint...")
    lora = create_lora_from_config(model_config, model)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    if "state_dict" in checkpoint:
        lora.load_weights(checkpoint["state_dict"])
    else:
        lora.load_weights(checkpoint)
    
    # Move LoRA weights to device
    for name, module in lora.net.lora_modules.items():
        module.lora_down.to(DEVICE)
        module.lora_up.to(DEVICE)
        if module.dora_mag is not None:
            module.dora_mag.to(DEVICE)
    
    lora.activate()
    print("✓ LoRA loaded and activated")
    
    print("\n[4/4] Extracting model parameters...")
    sample_rate = model_config.get("sample_rate", 44100)
    sample_size = model_config.get("sample_size", 265000)
    
    # Calculate latent size
    if model.pretransform is not None:
        latent_sample_size = sample_size // model.pretransform.downsampling_ratio
    else:
        latent_sample_size = sample_size
    
    print(f"✓ Sample rate: {sample_rate} Hz")
    print(f"✓ Sample size: {sample_size}")
    print(f"✓ Latent size: {latent_sample_size}")
    
    print("\n" + "=" * 80)
    print("MODEL READY - Waiting for requests...")
    print("=" * 80 + "\n")
    
except Exception as e:
    print(f"❌ ERROR during model loading: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# AUDIO GENERATION FUNCTION
# ============================================================================

def generate_audio(prompt, audio_length, cfg_scale, steps):
    """
    Generate audio using the loaded LoRA model.
    
    Args:
        prompt (str): Text description of footsteps
        audio_length (float): Duration in seconds
        cfg_scale (float): Classifier-free guidance scale (higher = more adherence to prompt)
        steps (int): Number of diffusion steps (higher = better quality, slower)
    
    Returns:
        torch.Tensor: Audio tensor [channels, samples]
    """
    # Validate inputs
    if audio_length <= 0 or audio_length > 47:
        raise ValueError(f"audio_length must be between 0 and 47 seconds, got {audio_length}")
    
    if cfg_scale < 1.0 or cfg_scale > 15.0:
        raise ValueError(f"cfg_scale must be between 1.0 and 15.0, got {cfg_scale}")
    
    if steps < 10 or steps > 200:
        raise ValueError(f"steps must be between 10 and 200, got {steps}")
    
    # Prepare batch metadata (list of dicts format)
    batch_metadata = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": audio_length
    }]
    
    # Generate seed and noise
    seed = np.random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    noise = torch.randn([1, model.io_channels, latent_sample_size], device=DEVICE)
    
    # Prepare conditioning
    conditioning_tensors = model.conditioner(batch_metadata, DEVICE)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    
    # Empty negative conditioning
    negative_conditioning_tensors = {}
    
    # Cast to model dtype
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v 
                          for k, v in conditioning_inputs.items()}
    
    # Sampling kwargs
    sampler_kwargs = {
        "sigma_min": 0.3,
        "sigma_max": 500,
        "rho": 1.0
    }
    
    # Run diffusion sampling
    diff_objective = model.diffusion_objective
    
    if diff_objective == "v":
        sampled = sample_k(
            model.model, noise, None, steps,
            **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors,
            cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=DEVICE
        )
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        sampler_kwargs_rf = {k: v for k, v in sampler_kwargs.items() 
                            if k not in ["sigma_min", "rho"]}
        sampled = sample_rf(
            model.model, noise, init_data=None, steps=steps,
            **sampler_kwargs_rf, **conditioning_inputs, **negative_conditioning_tensors,
            dist_shift=model.dist_shift, cfg_scale=cfg_scale, 
            batch_cfg=True, rescale_cfg=True, device=DEVICE
        )
    else:
        raise RuntimeError(f"Unknown diffusion objective: {diff_objective}")
    
    # Decode if using VAE
    if model.pretransform is not None:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)
    
    # Return first batch item
    return sampled[0]  # [channels, samples]

def audio_to_compressed_numpy(audio_tensor, sample_rate):
    """
    Convert audio tensor to compressed numpy array (base64 encoded).
    Uses np.savez_compressed for ~20x size reduction vs WAV.
    
    Args:
        audio_tensor (torch.Tensor): Audio tensor [channels, samples]
        sample_rate (int): Sample rate in Hz
    
    Returns:
        str: Base64 encoded compressed numpy array
    """
    audio_numpy = audio_tensor.cpu().numpy()
    
    # Save to compressed format in memory
    buffer = io.BytesIO()
    np.savez_compressed(buffer, audio=audio_numpy, sample_rate=sample_rate)
    
    # Get bytes and encode to base64
    buffer.seek(0)
    compressed_bytes = buffer.read()
    compressed_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    return compressed_base64

# ============================================================================
# RUNPOD HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "prompt": "heavy boots on marble floor",
            "audio_length": 6.0,
            "cfg_scale": 7.0,
            "steps": 100
        }
    }
    
    Returns:
    {
        "audio_base64": "SGVsbG8gd29ybGQ=...",
        "sample_rate": 44100,
        "duration_seconds": 6.0,
        "prompt": "heavy boots on marble floor",
        "status": "success",
        "metadata": {
            "generation_time_seconds": 25.3,
            "audio_shape": [2, 264600],
            "channels": 2,
            "num_samples": 264600,
            "max_amplitude": 0.543,
            "rms": 0.123,
            "peak_db": -5.3,
            "rms_db": -18.2,
            "cfg_scale": 7.0,
            "diffusion_steps": 100,
            "device": "cuda"
        }
    }
    
    Error response format:
    {
        "error": "Error message",
        "status": "failed",
        "traceback": "Full traceback..."
    }
    """
    try:
        print("\n" + "=" * 80)
        print("NEW REQUEST RECEIVED")
        print("=" * 80)
        
        # Extract input parameters with validation
        input_data = event.get("input", {})
        
        if not input_data:
            return {
                "error": "Missing 'input' object in request",
                "status": "failed"
            }
        
        prompt = input_data.get("prompt")
        audio_length = input_data.get("audio_length", 6.0)
        cfg_scale = input_data.get("cfg_scale", 7.0)
        steps = input_data.get("steps", 100)
        
        # Validate prompt
        if not prompt:
            return {
                "error": "Missing 'prompt' parameter",
                "status": "failed"
            }
        
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return {
                "error": "Invalid prompt: must be a non-empty string",
                "status": "failed"
            }
        
        # Type conversion and validation
        try:
            audio_length = float(audio_length)
            cfg_scale = float(cfg_scale)
            steps = int(steps)
        except (ValueError, TypeError) as e:
            return {
                "error": f"Invalid parameter type: {e}",
                "status": "failed"
            }
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Length: {audio_length}s")
        print(f"CFG Scale: {cfg_scale}")
        print(f"Steps: {steps}")
        print("\nGenerating audio...")
        
        # Generate audio with timing
        import time
        start_time = time.time()
        
        audio = generate_audio(
            prompt=prompt,
            audio_length=audio_length,
            cfg_scale=cfg_scale,
            steps=steps
        )
        
        generation_time = time.time() - start_time
        
        print(f"✓ Audio generated in {generation_time:.2f}s")
        
        # Convert to base64
        print("Encoding audio to base64...")
        audio_compressed = audio_to_compressed_numpy(audio, sample_rate)
        
        # Calculate metadata
        actual_duration = audio.shape[-1] / sample_rate
        max_amplitude = float(torch.max(torch.abs(audio)).item())
        rms = float(torch.sqrt(torch.mean(audio**2)).item())
        
        # Build response
        response = {
            "audio_base64": audio_compressed,
            "sample_rate": sample_rate,
            "duration_seconds": round(actual_duration, 2),
            "prompt": prompt,
            "status": "success",
            "metadata": {
                "generation_time_seconds": round(generation_time, 2),
                "audio_shape": list(audio.shape),
                "channels": audio.shape[0],
                "num_samples": audio.shape[-1],
                "max_amplitude": round(max_amplitude, 6),
                "rms": round(rms, 6),
                "peak_db": round(20 * np.log10(max_amplitude + 1e-10), 2),
                "rms_db": round(20 * np.log10(rms + 1e-10), 2),
                "cfg_scale": cfg_scale,
                "diffusion_steps": steps,
                "device": DEVICE
            }
        }
        
        print("✓ Response prepared")
        print(f"✓ Audio size: {len(audio_compressed)} bytes (base64)")
        print("=" * 80 + "\n")
        
        # Clear CUDA cache to prevent memory leaks
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return response
        
    except ValueError as e:
        # Parameter validation errors
        error_msg = f"Invalid parameters: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "error": error_msg,
            "status": "failed"
        }
        
    except Exception as e:
        # Unexpected errors
        error_msg = f"Error during generation: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "status": "failed"
        }

# ============================================================================
# START SERVERLESS WORKER
# ============================================================================

if __name__ == "__main__":
    print("Starting RunPod Serverless Worker...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print("=" * 80)
    
    runpod.serverless.start({"handler": handler})