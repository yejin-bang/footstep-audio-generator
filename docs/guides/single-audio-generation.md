# generate_single.py - Documentation

## Overview

Industry-standard single-prompt LoRA inference script for generating footstep audio. Designed for professional portfolio presentation and job interviews.

## What's New in This Version

### ✅ Phase Correlation Analysis
- Automatic stereo phase relationship analysis
- Industry-standard correlation thresholds
- Detects phase cancellation issues that could affect spatial processing
- Correlation ranges:
  - `< -0.7`: Severe phase issue (channels inverted)
  - `-0.7 to -0.3`: Mild phase issues
  - `> -0.3`: Good phase relationship

### ✅ Comprehensive Metadata
Returns detailed generation metadata including:
- **Timing**: Generation time in seconds and minutes
- **Audio Properties**: Sample rate, channels, duration, shape
- **Audio Statistics**: RMS, peak amplitude, dB levels
- **Phase Analysis**: Correlation coefficient and status
- **Generation Settings**: CFG scale, diffusion steps, device used

### ✅ Professional Metadata Display
Beautiful formatted output showing all generation details:
```
📝 Generation Summary:
   Prompt: 'heavy boots on marble floor'
   Checkpoint: best.ckpt
   Output: 20241108_152030_heavy_boots_on_marble_floor.wav

⏱️  Performance:
   Generation time: 45.23s (0.75 min)
   Device: cuda

🎵 Audio Properties:
   Sample rate: 44100 Hz
   Channels: 2
   Duration: 6.0s (target: 6.0s)
   Shape: [2, 264600]

📊 Audio Statistics:
   RMS: 0.123456 (-18.17 dB)
   Peak: 0.987654 (-0.11 dB)

🔊 Phase Analysis:
   Status: good
   Correlation: 0.856
   Phase relationship is healthy
```

## Usage

### Command Line

```bash
python generate_single.py \
  --prompt "heavy boots walking on marble floor" \
  --checkpoint /path/to/best.ckpt \
  --config /path/to/model_config_lora.json \
  --output ./outputs/
```

### As Python Module

```python
from generate_single import generate_footsteps

# Generate audio with full metadata
audio_tensor, sample_rate, output_path, metadata = generate_footsteps(
    prompt="high heels on concrete pavement",
    checkpoint_path="models/best.ckpt",
    config_path="config/model_config_lora.json",
    output_dir="outputs/",
    audio_length=6.0,
    cfg_scale=7.0,
    steps=100
)

# Access metadata
print(f"Generated in {metadata['generation_time_seconds']}s")
print(f"RMS: {metadata['rms']:.6f} ({metadata['rms_db']:.2f} dB)")
print(f"Phase correlation: {metadata['phase_info']['correlation']:.3f}")

# Pass to spatial processing
spatial_audio = apply_spatial_effects(audio_tensor, sample_rate, metadata)
```

## Return Values

The function returns a tuple of 4 values:

1. **`audio_tensor`** (torch.Tensor): Audio data [channels, samples]
2. **`sample_rate`** (int): Sample rate in Hz (44100)
3. **`output_path`** (Path): Path to saved WAV file
4. **`metadata`** (dict): Comprehensive generation metadata

### Metadata Dictionary Structure

```python
{
    # Core info
    "prompt": str,
    "checkpoint": str,
    "config": str,
    "output_path": str,
    "timestamp": str,
    
    # Timing
    "generation_time_seconds": float,
    "generation_time_minutes": float,
    
    # Audio properties
    "sample_rate": int,
    "audio_shape": list,
    "channels": int,
    "num_samples": int,
    "target_duration_seconds": float,
    "actual_duration_seconds": float,
    
    # Audio statistics
    "max_amplitude": float,
    "rms": float,
    "peak_db": float,
    "rms_db": float,
    
    # Phase analysis
    "phase_info": {
        "status": str,  # "good", "mild_phase_issue", or "severe_phase_issue"
        "correlation": float,  # -1.0 to 1.0
        "message": str
    },
    
    # Generation settings
    "cfg_scale": float,
    "diffusion_steps": int,
    "device": str,
    
    # Status
    "success": bool
}
```

## Phase Analysis Integration

### For Spatial Audio Processing

```python
# Generate audio
audio, sr, path, metadata = generate_footsteps(...)

# Check phase before spatial processing
phase_info = metadata['phase_info']

if phase_info['correlation'] < -0.3:
    print(f"⚠️ Warning: {phase_info['message']}")
    print("Consider phase correction before spatial processing")
    
# Apply spatial effects (binaural, panning, reverb, etc.)
spatial_audio = your_spatial_processor(audio, sr)
```

### Understanding Phase Correlation

**Pearson Correlation Coefficient** between left and right channels:
- **+1.0**: Perfect positive correlation (channels identical)
- **+0.7 to +1.0**: Good stereo image
- **0.0**: No correlation (independent channels)
- **-0.3 to 0.0**: Mild phase issues
- **-0.7 to -0.3**: Moderate phase cancellation
- **-1.0**: Perfect negative correlation (inverted phase)

## Technical Details

### Generation Pipeline

1. **Model Loading**: Loads Stable Audio Open base model from HuggingFace
2. **LoRA Integration**: Loads and activates your fine-tuned LoRA checkpoint
3. **Conditioning**: Prepares text embeddings from prompt
4. **Noise Generation**: Creates random latent noise with seed
5. **Diffusion Sampling**: 
   - Uses low-level `sample_k()` (v-objective) or `sample_rf()` (rectified flow)
   - Applies CFG (classifier-free guidance) for text conditioning
6. **VAE Decoding**: Decodes latents to waveform
7. **Quality Analysis**: Computes RMS, peak, phase correlation
8. **File Saving**: Saves as WAV with timestamp naming

### Device Handling

- **Automatic GPU detection** with fallback to CPU
- **Verbose device info**: Shows GPU name and VRAM
- **Memory cleanup**: Calls `torch.cuda.empty_cache()` after generation

## File Naming Convention

Output files use timestamp + sanitized prompt:
```
20241108_152030_heavy_boots_on_marble_floor.wav
20241108_152145_high_heels_clicking_on_concrete.wav
20241108_153220_barefoot_on_wooden_deck.wav
```

Benefits:
- ✅ Guaranteed unique (timestamp)
- ✅ Chronologically sortable
- ✅ Human-readable (includes prompt)
- ✅ Filesystem-safe (sanitized characters)

## Error Handling

The script follows industry-standard error handling:
- **Exit code 0**: Success
- **Exit code 1**: Error (with detailed message)
- **Path validation**: Checks all paths before loading
- **Clear error messages**: Explains what went wrong
- **Graceful degradation**: GPU → CPU fallback

## For Job Interviews

### What This Script Demonstrates

1. **Production-Ready Code**
   - Type hints throughout
   - Comprehensive docstrings
   - Error handling with exit codes
   - Clean separation of concerns

2. **Deep Technical Understanding**
   - Low-level diffusion sampling (not black-box wrapper)
   - Proper LoRA integration and activation
   - Phase correlation analysis (audio engineering knowledge)
   - VAE encoding/decoding pipeline

3. **Industry Standards**
   - Metadata tracking for reproducibility
   - Quality metrics (RMS, peak, dB)
   - Professional file naming
   - GPU/CPU abstraction

4. **Pipeline Design**
   - Returns tensor for further processing
   - Always saves file (debugging/auditing)
   - Comprehensive metadata (logging/analysis)
   - Clean API (easy integration)

### Key Points to Mention

- **"I implemented low-level diffusion sampling to ensure LoRA compatibility"**
  - Shows understanding of the generation pipeline
  - Not just using high-level wrappers blindly

- **"I added phase correlation analysis for quality monitoring"**
  - Shows audio engineering knowledge
  - Important for spatial audio processing

- **"The script returns both the file and tensor for pipeline flexibility"**
  - Shows API design thinking
  - Supports both debugging (file) and processing (tensor)

- **"Comprehensive metadata enables reproducibility and analysis"**
  - Shows production/research mindset
  - All generation parameters tracked

## Comparison to Previous Versions

| Feature | Old Version | New Version |
|---------|------------|-------------|
| Metadata | ❌ None | ✅ Comprehensive dict |
| Phase Analysis | ❌ No | ✅ Automatic |
| Timing Info | ❌ No | ✅ Seconds + minutes |
| Audio Stats | ❌ No | ✅ RMS, peak, dB |
| Return Values | 3 items | 4 items (+ metadata) |
| Console Output | Basic | Formatted with emojis |

## Dependencies

```
torch
numpy
torchaudio
stable-audio-tools
LoRAW (custom framework)
```

## Path Configuration

Update these paths in the script (lines 23-24):
```python
LORAW_PATH = "/workspace/LoRAW"  # Your LoRAW installation
STABLE_AUDIO_PATH = "/workspace/stable-audio-tools"  # Your stable-audio-tools installation
```

## Example Output

```
================================================================================
LoRA Footstep Generation - Single Prompt Inference
================================================================================

✓ GPU detected: NVIDIA RTX A5000
  VRAM: 24.0 GB
✓ Checkpoint: best.ckpt
✓ Config: model_config_lora.json
✓ Output directory: outputs/

--------------------------------------------------------------------------------
Loading base model...
--------------------------------------------------------------------------------
✓ Base model loaded
✓ Model config loaded

--------------------------------------------------------------------------------
Loading LoRA checkpoint...
--------------------------------------------------------------------------------
✓ LoRA weights loaded
✓ LoRA activated
  Sample rate: 44100 Hz
  Sample size: 264600

--------------------------------------------------------------------------------
Generating audio...
--------------------------------------------------------------------------------
  Prompt: 'heavy boots walking on marble floor'
  Length: 6.0s
  CFG Scale: 7.0
  Steps: 100
  Diffusion objective: v
  Seed: 1234567890
✓ Audio generated successfully
  Shape: [2, 264600]
  Duration: 6.00s
✓ Audio saved: 20241108_152030_heavy_boots_walking_on_marble_floor.wav

================================================================================
Generation Complete - Metadata Summary
================================================================================

📝 Generation Summary:
   Prompt: 'heavy boots walking on marble floor'
   Checkpoint: best.ckpt
   Output: 20241108_152030_heavy_boots_walking_on_marble_floor.wav

⏱️  Performance:
   Generation time: 45.23s (0.75 min)
   Device: cuda

🎵 Audio Properties:
   Sample rate: 44100 Hz
   Channels: 2
   Duration: 6.0s (target: 6.0s)
   Shape: [2, 264600]

📊 Audio Statistics:
   RMS: 0.123456 (-18.17 dB)
   Peak: 0.987654 (-0.11 dB)

🔊 Phase Analysis:
   Status: good
   Correlation: 0.856
   Phase relationship is healthy

⚙️  Generation Settings:
   CFG scale: 7.0
   Diffusion steps: 100

✓ Audio tensor returned for spatial audio processing
```

## License

This script is part of Yejin Bang's portfolio project for professional footstep sound generation using LoRA fine-tuning on Stable Audio Open.
