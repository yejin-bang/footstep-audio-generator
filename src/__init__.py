"""
Footstep Audio Generation Pipeline - Core Components

This package contains the main pipeline components for generating
spatialized footstep audio from video input.

Components:
- video_validator: Video file validation and metadata extraction
- footstep_detector: Footstep detection with spatial data extraction
- scene_analyzer: CLIP-based scene analysis and prompt generation
- audio_generator: Audio generation via RunPod or local GPU
- spatial_audio_processor: Spatial audio processing and mixing
- main_pipeline: End-to-end pipeline orchestration
"""

__version__ = "1.0.0"

# Only import when accessed to avoid dependency issues
# This allows using utils/config without installing all pipeline dependencies
def __getattr__(name):
    """Lazy import main components."""
    if name == "VideoValidator":
        from .pipeline.video_validator import VideoValidator
        return VideoValidator
    elif name == "FootstepDetector":
        from .pipeline.footstep_detector import FootstepDetector
        return FootstepDetector
    elif name == "DetectorConfig":
        from .pipeline.footstep_detector import DetectorConfig
        return DetectorConfig
    elif name == "SceneAnalyzer":
        from .pipeline.scene_analyzer import SceneAnalyzer
        return SceneAnalyzer
    elif name == "generate_footsteps":
        from .pipeline.audio_generator import generate_footsteps
        return generate_footsteps
    elif name == "SpatialAudioProcessor":
        from .pipeline.spatial_audio_processor import SpatialAudioProcessor
        return SpatialAudioProcessor
    elif name == "FootstepAudioPipeline":
        from .pipeline.main_pipeline import FootstepAudioPipeline
        return FootstepAudioPipeline
    elif name == "PipelineConfig":
        from .pipeline.main_pipeline import PipelineConfig
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
