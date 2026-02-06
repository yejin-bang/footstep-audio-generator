#!/usr/bin/env python3
"""
RunPod Serverless Client
"""

import requests
import base64
import argparse
import sys
import io
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

def sanitize_filename(prompt: str, max_length: int = 50) -> str:
    """Convert prompt to valid filename."""
    import re
    from datetime import datetime
    
    # Replace spaces with underscores
    filename = prompt.replace(" ", "_")
    
    # Remove special characters (keep only alphanumeric, underscore, hyphen)
    filename = re.sub(r'[^\w\-]', '', filename)
    
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # Remove trailing underscores/hyphens
    filename = filename.rstrip("_-")
    
    return filename

def get_output_path(prompt: str, user_output: str = None) -> Path:
    """
    Generate output path from prompt or use user-provided path.
    Adds timestamp if file already exists.
    """
    from datetime import datetime
    
    if user_output:
        output_path = Path(user_output)
    else:
        # Auto-generate from prompt
        base_name = sanitize_filename(prompt)
        output_path = Path(f"{base_name}.wav")
    
    # Check if file exists and add timestamp if needed
    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"
        print(f"⚠️  File exists, using timestamped name: {output_path.name}")
    
    return output_path

def call_serverless_endpoint(
    endpoint_url: str,
    api_key: str,
    prompt: str,
    audio_length: float = 6.0,
    cfg_scale: float = 7.0,
    steps: int = 100,
    poll_interval: float = 2.0,
    timeout: int = None
) -> dict:
    """Call the RunPod serverless endpoint (ASYNC mode with polling)."""
    # Get timeout from .env if not provided
    if timeout is None:
        timeout = int(os.getenv("RUNPOD_TIMEOUT", "70"))
    
    print("=" * 80)
    print("Calling RunPod Serverless Endpoint (Async)")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print(f"Length: {audio_length}s")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Steps: {steps}")
    print()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "audio_length": audio_length,
            "cfg_scale": cfg_scale,
            "steps": steps
        }
    }
    
    # Step 1: Submit job
    print("Submitting job...")
    response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
    
    if response.status_code != 200:
        print(f"✗ Error: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
    
    result = response.json()
    
    if "id" not in result:
        print(f"✗ Error: No job ID in response")
        print(f"Response: {result}")
        sys.exit(1)
    
    job_id = result["id"]
    print(f"✓ Job submitted: {job_id}")
    
    # Step 2: Poll for completion
    base_url = endpoint_url.replace('/run', '')
    status_url = f"{base_url}/status/{job_id}"
    
    print(f"Polling for results (timeout: {timeout}s)...")
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        # Check client-side timeout
        if elapsed > timeout:
            print(f"\n⚠️  Client timeout reached ({timeout}s)")
            print(f"Cancelling job {job_id} to stop worker...")
            
            # Cancel the job via RunPod API
            cancel_url = f"{base_url}/cancel/{job_id}"
            try:
                cancel_response = requests.post(cancel_url, headers=headers, timeout=5)
                if cancel_response.status_code == 200:
                    print(f"✓ Job cancelled successfully")
                else:
                    print(f"⚠️  Cancel request returned: {cancel_response.status_code}")
            except Exception as e:
                print(f"⚠️  Failed to cancel job: {e}")
            
            print(f"✗ Timeout after {timeout}s - worker has been stopped")
            sys.exit(1)
        
        status_response = requests.get(status_url, headers=headers, timeout=10)
        
        if status_response.status_code != 200:
            print(f"✗ Status check failed: HTTP {status_response.status_code}")
            sys.exit(1)
        
        status_result = status_response.json()
        status = status_result.get("status")
        
        if status == "IN_QUEUE":
            print(f"  ⏳ In queue... ({elapsed:.1f}s)")
        elif status == "IN_PROGRESS":
            print(f"  🔄 Processing... ({elapsed:.1f}s)")
        elif status == "COMPLETED":
            print(f"✓ Completed in {elapsed:.1f}s")
            print()
            return status_result
        elif status == "FAILED":
            print(f"✗ Job failed: {status_result.get('error', 'Unknown error')}")
            sys.exit(1)
        else:
            print(f"  ⚠️  Unknown status: {status}")
        
        time.sleep(poll_interval)

def save_audio_from_base64(audio_base64: str, output_path: Path, sample_rate: int):
    """Decode base64 NPZ and save as proper WAV file."""
    
    print(f"Decoding and saving audio to: {output_path}")
    
    # Decode base64 to bytes
    audio_bytes = base64.b64decode(audio_base64)
    
    # Load NPZ from bytes
    buf = io.BytesIO(audio_bytes)
    data = np.load(buf)
    
    audio = data["audio"]  # Shape: [channels, samples]
    sr = int(data["sample_rate"])
    
    print(f"  Audio shape: {audio.shape}")
    print(f"  Sample rate: {sr} Hz")
    
    # soundfile expects [samples, channels]
    audio_transposed = audio.T
    
    # Save as WAV
    sf.write(output_path, audio_transposed, sr)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Audio saved ({file_size_mb:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(
        description="Call RunPod serverless footstep generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-generate output filename from prompt
  python runpod_client.py --prompt "boots on marble"
  # Creates: boots_on_marble.wav
  
  # If file exists, timestamp is added automatically
  python runpod_client.py --prompt "boots on marble"
  # Creates: boots_on_marble_20241115_143022.wav
  
  # Override with custom filename
  python runpod_client.py --prompt "boots on marble" --output footstep.wav
  
  # Using .env credentials with custom parameters
  python runpod_client.py \\
    --prompt "high heels on wood" \\
    --length 10.0 \\
    --steps 150
  
  # Override .env with CLI arguments
  python runpod_client.py \\
    --endpoint https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \\
    --api-key YOUR_API_KEY \\
    --prompt "boots on marble"
        """
    )
    
    parser.add_argument("--endpoint", type=str, default=None,
                       help="RunPod endpoint URL (or set RUNPOD_ENDPOINT_URL in .env)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Your RunPod API key (or set RUNPOD_API_KEY in .env)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text description of footsteps")
    parser.add_argument("--output", type=str, required=False, default=None,
                       help="Output audio file path (.wav) - auto-generated from prompt if not provided")
    parser.add_argument("--length", type=float, default=6.0,
                       help="Audio length in seconds (default: 6.0)")
    parser.add_argument("--cfg", type=float, default=7.0,
                       help="CFG scale (default: 7.0)")
    parser.add_argument("--steps", type=int, default=100,
                       help="Diffusion steps (default: 100)")
    
    args = parser.parse_args()
    
    # Get credentials from .env if not provided via CLI
    endpoint_url = args.endpoint or os.getenv("RUNPOD_ENDPOINT_URL")
    api_key = args.api_key or os.getenv("RUNPOD_API_KEY")
    
    # Validate credentials
    if not endpoint_url:
        print("Error: RunPod endpoint URL not found!")
        print("Either:")
        print("  1. Set RUNPOD_ENDPOINT_URL in .env file")
        print("  2. Pass --endpoint on command line")
        sys.exit(1)
    
    if not api_key:
        print("Error: RunPod API key not found!")
        print("Either:")
        print("  1. Set RUNPOD_API_KEY in .env file")
        print("  2. Pass --api-key on command line")
        sys.exit(1)
    
    print(f"✓ Using endpoint: {endpoint_url}")
    print(f"✓ API key loaded: {api_key[:8]}..." if len(api_key) > 8 else "✓ API key loaded")
    print()
    
    # Generate or validate output path
    output_path = get_output_path(args.prompt, args.output)
    
    # Validate output path extension
    if output_path.suffix != ".wav":
        print("Error: Output file must have .wav extension")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output file: {output_path}")
    print()
    
    try:
        # Call serverless endpoint
        result = call_serverless_endpoint(
            endpoint_url=endpoint_url,
            api_key=api_key,
            prompt=args.prompt,
            audio_length=args.length,
            cfg_scale=args.cfg,
            steps=args.steps
        )
        
        # Extract output
        output = result.get("output", {})
        
        if "error" in output:
            print(f"✗ Serverless function error: {output['error']}")
            if "traceback" in output:
                print("\nTraceback:")
                print(output["traceback"])
            sys.exit(1)
        
        audio_base64 = output.get("audio_base64")
        sample_rate = output.get("sample_rate")
        metadata = output.get("metadata", {})
        
        if not audio_base64:
            print("✗ Error: No audio in response")
            sys.exit(1)
        
        # Save audio
        save_audio_from_base64(audio_base64, output_path, sample_rate)
        
        # Print metadata
        print()
        print("=" * 80)
        print("Generation Metadata")
        print("=" * 80)
        print(f"Generation time: {metadata.get('generation_time_seconds')}s")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {metadata.get('duration_seconds')}s")
        print(f"Channels: {metadata.get('channels')}")
        print(f"RMS: {metadata.get('rms_db')} dB")
        print(f"Peak: {metadata.get('peak_db')} dB")
        print(f"Device: {metadata.get('device')}")
        print()
        print("✅ Complete! You can now play the file:")
        print(f"   {output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        print("⚠️  Note: RunPod worker may still be processing.")
        print("⚠️  The job will auto-cancel after 50 seconds if still running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: Network request failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()