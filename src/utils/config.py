"""
Centralized Configuration for Footstep Audio Pipeline

Provides centralized path management and configuration constants.
Eliminates hardcoded paths throughout the codebase.

Usage:
    from src.utils.config import CONFIG_DIR, TEST_VIDEOS_DIR, get_test_video

    # Use predefined paths
    caption_config = CONFIG_DIR / "caption_config.json"

    # Get test resources
    test_video = get_test_video("walk1.mp4")
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# ============================================================================
# Project Structure
# ============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# Utilities directory (now part of src/)
UTILS_DIR = SRC_DIR / "utils"

# ============================================================================
# Configuration Files
# ============================================================================

CONFIG_DIR = PROJECT_ROOT / "config"
CAPTION_CONFIG_PATH = CONFIG_DIR / "caption_config.json"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config_lora.json"
SCENE_CONFIG_PATH = CONFIG_DIR / "scene_config.json"

# ============================================================================
# Model Files
# ============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
LORA_CHECKPOINT_PATH = MODELS_DIR / "best.ckpt"

# ============================================================================
# Test Resources
# ============================================================================

# Test data is now organized under data/ directory
DATA_DIR = PROJECT_ROOT / "data"
TEST_VIDEOS_DIR = DATA_DIR / "videos"
TEST_AUDIOS_DIR = DATA_DIR / "audio"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# ============================================================================
# Output Directories
# ============================================================================

# Default output directories (created automatically)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PIPELINE_OUTPUTS_DIR = OUTPUTS_DIR / "pipeline"
GENERATED_OUTPUTS_DIR = OUTPUTS_DIR / "generated"
LOGS_DIR = PROJECT_ROOT / "logs"


def get_video_output_dir(video_path: str) -> Path:
    """
    Get output directory for a specific video with timestamp.

    Args:
        video_path: Path to video file (e.g., "walk4.mp4" or "/path/to/walk4.mp4")

    Returns:
        Path to video-specific output directory with timestamp (e.g., outputs/walk4_outputs_20250121_143022/)

    Example:
        >>> output_dir = get_video_output_dir("walk4.mp4")
        >>> print(output_dir)
        /path/to/project/outputs/walk4_outputs_20250121_143022
    """
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_DIR / f"{video_name}_outputs_{timestamp}"

# ============================================================================
# Environment Variables
# ============================================================================

# RunPod configuration (from .env)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL")
RUNPOD_TIMEOUT = int(os.getenv("RUNPOD_TIMEOUT", "300"))

# LoRA paths for local GPU mode (optional)
LORAW_PATH = os.getenv("LORAW_PATH")
STABLE_AUDIO_PATH = os.getenv("STABLE_AUDIO_PATH")

# ============================================================================
# Helper Functions
# ============================================================================

def get_test_video(filename: str) -> Path:
    """
    Get path to a test video file.

    Args:
        filename: Name of test video (e.g., "walk1.mp4")

    Returns:
        Full path to test video

    Raises:
        FileNotFoundError: If test video doesn't exist

    Example:
        >>> video_path = get_test_video("walk1.mp4")
        >>> print(video_path)
        /path/to/project/data/videos/walk1.mp4
    """
    path = TEST_VIDEOS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Test video not found: {path}\n"
            f"Available videos: {list_test_videos()}"
        )
    return path


def get_test_audio(filename: str) -> Path:
    """
    Get path to a test audio file.

    Args:
        filename: Name of test audio (e.g., "footsteps.wav")

    Returns:
        Full path to test audio

    Raises:
        FileNotFoundError: If test audio doesn't exist
    """
    path = TEST_AUDIOS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Test audio not found: {path}\n"
            f"Available audios: {list_test_audios()}"
        )
    return path


def list_test_videos() -> list:
    """
    List all available test videos.

    Returns:
        List of test video filenames
    """
    if not TEST_VIDEOS_DIR.exists():
        return []
    return [f.name for f in TEST_VIDEOS_DIR.glob("*.mp4")]


def list_test_audios() -> list:
    """
    List all available test audio files.

    Returns:
        List of test audio filenames
    """
    if not TEST_AUDIOS_DIR.exists():
        return []
    return [f.name for f in TEST_AUDIOS_DIR.glob("*.wav")]


def ensure_output_dirs() -> None:
    """
    Create output directories if they don't exist.

    Call this at the start of the pipeline to ensure
    all output directories are ready.
    """
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get configuration value from environment variables.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Configuration value or default

    Example:
        >>> api_key = get_config_value("RUNPOD_API_KEY")
    """
    return os.getenv(key, default)


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config() -> dict:
    """
    Validate that all required configuration is present.

    Returns:
        Dictionary with validation results

    Example:
        >>> validation = validate_config()
        >>> if not validation['valid']:
        ...     print(validation['errors'])
    """
    errors = []
    warnings = []

    # Check required directories
    if not CONFIG_DIR.exists():
        errors.append(f"Config directory not found: {CONFIG_DIR}")

    if not CAPTION_CONFIG_PATH.exists():
        errors.append(f"Caption config not found: {CAPTION_CONFIG_PATH}")

    # Check optional resources
    if not TEST_VIDEOS_DIR.exists():
        warnings.append(f"Test videos directory not found: {TEST_VIDEOS_DIR}")

    if not TEST_AUDIOS_DIR.exists():
        warnings.append(f"Test audios directory not found: {TEST_AUDIOS_DIR}")

    # Check RunPod configuration
    if not RUNPOD_API_KEY:
        warnings.append("RUNPOD_API_KEY not set (required for RunPod backend)")

    if not RUNPOD_ENDPOINT_URL:
        warnings.append("RUNPOD_ENDPOINT_URL not set (required for RunPod backend)")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


# ============================================================================
# Print Configuration Summary
# ============================================================================

if __name__ == "__main__":
    """Print configuration summary for debugging."""
    print("=" * 80)
    print("FOOTSTEP AUDIO PIPELINE - CONFIGURATION")
    print("=" * 80)
    print()

    print("Project Structure:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  SRC_DIR: {SRC_DIR}")
    print(f"  CONFIG_DIR: {CONFIG_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")
    print()

    print("Test Resources:")
    print(f"  TEST_VIDEOS_DIR: {TEST_VIDEOS_DIR}")
    print(f"  Available videos: {len(list_test_videos())}")
    print(f"  TEST_AUDIOS_DIR: {TEST_AUDIOS_DIR}")
    print(f"  Available audios: {len(list_test_audios())}")
    print()

    print("Output Directories:")
    print(f"  PIPELINE_OUTPUTS_DIR: {PIPELINE_OUTPUTS_DIR}")
    print(f"  GENERATED_OUTPUTS_DIR: {GENERATED_OUTPUTS_DIR}")
    print(f"  LOGS_DIR: {LOGS_DIR}")
    print()

    print("Environment Variables:")
    print(f"  RUNPOD_API_KEY: {'✓ Set' if RUNPOD_API_KEY else '✗ Not set'}")
    print(f"  RUNPOD_ENDPOINT_URL: {'✓ Set' if RUNPOD_ENDPOINT_URL else '✗ Not set'}")
    print(f"  RUNPOD_TIMEOUT: {RUNPOD_TIMEOUT}s")
    print()

    print("Validation:")
    validation = validate_config()
    if validation['valid']:
        print("  ✓ Configuration is valid")
    else:
        print("  ✗ Configuration has errors:")
        for error in validation['errors']:
            print(f"    - {error}")

    if validation['warnings']:
        print("\n  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    print()
    print("=" * 80)
