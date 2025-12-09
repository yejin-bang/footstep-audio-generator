import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2

from .logger import setupt_logging, get_logger

logger = get_logger(__name__)

from .footstep_detector import SimpleFootstepDetector, SimpleDetectorConfig
from .scene_analyzer import SceneAnalyzer 
from .spatial_audio_processor import SpatialAudioProcessor
from .video_merger import merge_audio_video, check_ffmpeg_installed


try:
    from .audio_generator import generate_footsteps
    AUDIO_GENERATOR_AVAILABLE = True
except ImportError:
    print(f" Warning: audio_generator.py not found")
    AUDIO_GENERATOR_AVAILABLE = False

@dataclass
class PipelineConfig:
    bakend: str = "runpod"
    audio_variations: int = 1
    audio_length: float = 66.0
    cgf_scale: float = 7.0
    steps: int = 100

    target_fps: int = 10
    confidence_threshold: float = 0.7

    scene_analysis_seed= 42
    
    sample_rate =44100
    fade_duration = 0.01
    min_attenuation_db = -20
    max_attenuation_db = 0

    output_dir: "./pipeline_outputs"
    save_intermediates = True
    create_visualization = True
    merge_vdieo =True

    def __post_init__(self):
        from .audio_backends import list_backends

        available_backends = list_backends()
        if self.backend not in available_backends:
            raise ValueError(
                f"backend must be one of {available_backends}"
            )
        if self.audio_variations < 1:
            raise ValueError(f"audio_variations must be >=1")
        

class FootstepAudioPipeline:
    def __init__(self, config: Optional[PipelineConfig]):
        self.config = config if config is not None else PipelineConfig()

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.footstep_detector = SimpleFootstepDetector(
            
        )

