"""
Video and Audio Merger

Combines original video with generated footstep audio using ffmpeg.
Creates a new video file with the audio track replaced.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple
import shutil


def check_ffmpeg_installed() -> bool:
    """
    Check if ffmpeg is installed and available in PATH.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    return shutil.which("ffmpeg") is not None


def merge_audio_video(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Merge audio track with video using ffmpeg.

    Args:
        video_path: Path to input video file
        audio_path: Path to generated audio file (WAV)
        output_path: Path for output video (overrides output_dir if provided)
        output_dir: Directory for output video (default: video's parent directory)
        verbose: Print progress information

    Returns:
        Tuple of (success: bool, output_path: str)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("VIDEO & AUDIO MERGER")
        print("=" * 80)

    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        raise FileNotFoundError(
            "ffmpeg is not installed."
        )

    # Validate input files
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Generate output path if not provided
    if output_path is None:
        # Use output_dir if provided, otherwise use video's parent directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_path = output_dir / f"{video_path.stem}_with_footsteps{video_path.suffix}"
        else:
            output_path = video_path.parent / f"{video_path.stem}_with_footsteps{video_path.suffix}"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Input video: {video_path.name}")
        print(f"Input audio: {audio_path.name}")
        print(f"Output video: {output_path.name}")
        print()
        print("Merging with ffmpeg...")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),      # Input video
        "-i", str(audio_path),      # Input audio
        "-c:v", "copy",             # Copy video (no re-encoding)
        "-c:a", "aac",              # Encode audio to AAC
        "-map", "0:v",              # Use video from first input
        "-map", "1:a",              # Use audio from second input
        "-shortest",                # Match shortest stream
        "-y",                       # Overwrite if exists
        str(output_path)
    ]

    try:
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if verbose:
            print(f"✓ Successfully merged video and audio!")
            print(f"  Output: {output_path}")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {file_size_mb:.1f} MB")
            print("=" * 80)

        return True, str(output_path)

    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with error:\n{e.stderr}"
        if verbose:
            print(f"✗ Merge failed!")
            print(error_msg)
            print("=" * 80)
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if verbose:
            print(f"✗ Merge failed!")
            print(error_msg)
            print("=" * 80)
        return False, error_msg


# CLI for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge video with generated footstep audio using ffmpeg"
    )
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("audio", type=str, help="Generated audio file (WAV)")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output video file (default: {video}_with_footsteps.mp4)")

    args = parser.parse_args()

    try:
        success, output = merge_audio_video(
            args.video,
            args.audio,
            args.output,
            verbose=True
        )

        if success:
            print(f"\n✅ Success! Merged video: {output}")
            exit(0)
        else:
            print(f"\n❌ Failed: {output}")
            exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
