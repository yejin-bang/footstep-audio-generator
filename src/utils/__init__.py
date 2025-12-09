"""
Footstep Audio Generation Pipeline - Utility Modules

This package contains infrastructure utilities for the footstep audio generation pipeline.

Components:
- config: Centralized configuration and path management
- logger: Logging system for the pipeline
- pose_extractor: MediaPipe pose landmark extraction
- runpod_api: RunPod serverless API client (async with polling)
- runpod_client: Command-line RunPod client
- video_merger: ffmpeg-based audio-video merging utility
"""

__version__ = "1.0.0"

# Lazy imports - only import when accessed to avoid dependency issues
# This allows using config/logger without installing opencv, mediapipe, etc.
def __getattr__(name):
    """Lazy import utilities."""
    # Config module (no dependencies)
    if name in ["PROJECT_ROOT", "CONFIG_DIR", "CAPTION_CONFIG_PATH", "MODEL_CONFIG_PATH",
                "SCENE_CONFIG_PATH", "DATA_DIR", "TEST_VIDEOS_DIR", "TEST_AUDIOS_DIR",
                "OUTPUTS_DIR", "PIPELINE_OUTPUTS_DIR", "GENERATED_OUTPUTS_DIR",
                "get_test_video", "get_test_audio", "list_test_videos", "list_test_audios",
                "ensure_output_dirs"]:
        from . import config
        return getattr(config, name)

    # Logger module (no heavy dependencies)
    elif name in ["setup_logging", "get_logger"]:
        from . import logger
        return getattr(logger, name)

    # Pose extractor (requires opencv, mediapipe)
    elif name == "PoseExtractor":
        from .pose_extractor import PoseExtractor
        return PoseExtractor

    # RunPod API (requires requests)
    elif name in ["RunPodClient", "RunPodError", "generate_footstep_audio"]:
        from . import runpod_api
        return getattr(runpod_api, name)

    # Video merger (requires ffmpeg)
    elif name in ["merge_audio_video", "check_ffmpeg_installed"]:
        from . import video_merger
        return getattr(video_merger, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Pose extraction
    "PoseExtractor",
    # RunPod API
    "RunPodClient",
    "RunPodError",
    "generate_footstep_audio",
    # Video merging
    "merge_audio_video",
    "check_ffmpeg_installed",
    # Logging
    "setup_logging",
    "get_logger",
    # Configuration - Paths
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "CAPTION_CONFIG_PATH",
    "MODEL_CONFIG_PATH",
    "SCENE_CONFIG_PATH",
    "DATA_DIR",
    "TEST_VIDEOS_DIR",
    "TEST_AUDIOS_DIR",
    "OUTPUTS_DIR",
    "PIPELINE_OUTPUTS_DIR",
    "GENERATED_OUTPUTS_DIR",
    # Configuration - Helpers
    "get_test_video",
    "get_test_audio",
    "list_test_videos",
    "list_test_audios",
    "ensure_output_dirs",
]
