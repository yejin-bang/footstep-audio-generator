#!/usr/bin/env python3
"""
Audio Generation with Pluggable Backends

Wrapper module for generating footstep audio using different backends
(RunPod, local GPU, mock, etc.). Provides a simple interface for the pipeline.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional
import numpy as np

from ..audio_backends import get_backend, list_backends


# Utility functions
def create_safe_filename(text: str, max_length: int = 100) -> str:
    """Create filesystem-safe filename from text."""
    safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)
    safe = safe.replace(' ', '_').lower()
    return safe[:max_length]



# Main Generation Function
def generate_footsteps(
    prompt: str,
    output_dir: str = "outputs/generated",
    audio_length: float = 6.0,
    cfg_scale: float = 7.0,
    steps: int = 100,
    backend: str = "runpod",
    **backend_kwargs
) -> Tuple[np.ndarray, int, Path, Dict]:
    """
    Generate footstep audio using the specified backend.

    This is the main interface used by the pipeline. It abstracts away
    the backend implementation details.

    Args:
        prompt: Text description of the footsteps to generate
        output_dir: Directory to save generated audio (default: "outputs/generated")
        audio_length: Duration of audio in seconds
        cfg_scale: Classifier-free guidance scale
        steps: Number of diffusion steps
        backend: Backend to use ("runpod", "mock", etc.)
        **backend_kwargs: Additional backend-specific arguments

    Returns:
        Tuple of (audio_array, sample_rate, output_path, metadata)
    """
    import time

    print(f"🎵 Generating audio using '{backend}' backend...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Length: {audio_length}s, CFG: {cfg_scale}, Steps: {steps}")

    start_time = time.time()

    # Get the backend
    try:
        audio_backend = get_backend(backend, **backend_kwargs)
    except ValueError as e:
        available = ", ".join(list_backends())
        raise ValueError(f"{e}\nAvailable backends: {available}")

    # Generate audio
    try:
        audio, sample_rate, gen_metadata = audio_backend.generate(
            prompt=prompt,
            audio_length=audio_length,
            cfg_scale=cfg_scale,
            steps=steps
        )
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {e}") from e

    end_time = time.time()
    generation_time = end_time - start_time

    # Prepare output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{create_safe_filename(prompt)}.wav"
    output_path = output_dir / filename

    # Save audio
    import soundfile as sf
    sf.write(output_path, audio.T, sample_rate)

    channels, samples = audio.shape
    duration = samples / sample_rate

    print(f"✅ Audio saved: {output_path} ({channels}ch, {duration:.2f}s, {sample_rate}Hz)")

    # Build comprehensive metadata
    metadata = {
        "prompt": prompt,
        "backend": backend,
        "output_path": str(output_path),
        "generation_time_seconds": round(generation_time, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "num_samples": samples,
        "duration_seconds": round(duration, 2),
        "cfg_scale": cfg_scale,
        "diffusion_steps": steps,
        "success": True,
        **gen_metadata  # Merge backend-specific metadata
    }

    return audio, sample_rate, output_path, metadata

# CLI
def main():
    """Command-line interface for audio generation."""
    parser = argparse.ArgumentParser(
        description="Footstep audio generation with pluggable backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available backends: {', '.join(list_backends())}

Examples:
  # Generate using RunPod (default)
  python -m src.audio_generator --prompt "boots on marble"

  # Generate using mock backend for testing
  python -m src.audio_generator --prompt "boots on marble" --backend mock

  # Custom parameters
  python -m src.audio_generator \\
    --prompt "high heels on wood" \\
    --length 10.0 \\
    --cfg 8.0 \\
    --steps 150
        """
    )

    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for footsteps")
    parser.add_argument("--output", type=str, default="outputs/generated",
                       help="Output directory (default: outputs/generated)")
    parser.add_argument("--length", type=float, default=6.0,
                       help="Audio length in seconds (default: 6.0)")
    parser.add_argument("--cfg", type=float, default=7.0,
                       help="CFG scale (default: 7.0)")
    parser.add_argument("--steps", type=int, default=100,
                       help="Diffusion steps (default: 100)")
    parser.add_argument("--backend", type=str, default="runpod",
                       choices=list_backends(),
                       help="Audio generation backend (default: runpod)")

    args = parser.parse_args()

    try:
        audio, sr, path, meta = generate_footsteps(
            prompt=args.prompt,
            output_dir=args.output,
            audio_length=args.length,
            cfg_scale=args.cfg,
            steps=args.steps,
            backend=args.backend
        )

        print("\n" + "=" * 80)
        print("Generation Metadata:")
        print("=" * 80)
        print(json.dumps(meta, indent=2))
        print("\n✅ Success! Audio saved to:", path)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
