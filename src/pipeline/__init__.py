"""
Footstep Audio Generation Pipeline - Core Components

This package contains the main pipeline components for generating
spatialized footstep audio from video input.

Pipeline Flow:
1. Video Validation - Extract metadata (FPS, duration, resolution)
2. Footstep Detection - MediaPipe pose + Hip-heel distances + Peak detection
3. Scene Analysis - CLIP environment classification + Generate prompts
4. Audio Generation - RunPod/Local GPU + LoRA-tuned Stable Audio
5. Spatial Processing - Panning + Distance attenuation + Peak alignment
6. Finalization - Match video duration + Export WAV
"""

__version__ = "1.0.0"

# only import when accessed to avoid dependency issues
def __getattr__(name):
    """Lazy import main components."""
    if name == "VideoValidator":
        from .video_validator import VideoValidator
        return VideoValidator
    elif name == "FootstepDetector":
        from .footstep_detector import FootstepDetector
        return FootstepDetector
    elif name == "DetectorConfig":
        from .footstep_detector import DetectorConfig
        return DetectorConfig
    elif name == "SceneAnalyzer":
        from .scene_analyzer import SceneAnalyzer
        return SceneAnalyzer
    elif name == "generate_footsteps":
        from .audio_generator import generate_footsteps
        return generate_footsteps
    elif name == "SpatialAudioProcessor":
        from .spatial_audio_processor import SpatialAudioProcessor
        return SpatialAudioProcessor
    elif name == "FootstepAudioPipeline":
        from .main_pipeline import FootstepAudioPipeline
        return FootstepAudioPipeline
    elif name == "PipelineConfig":
        from .main_pipeline import PipelineConfig
        return PipelineConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "VideoValidator",
    "FootstepDetector",
    "DetectorConfig",
    "SceneAnalyzer",
    "generate_footsteps",
    "SpatialAudioProcessor",
    "FootstepAudioPipeline",
    "PipelineConfig",
]
