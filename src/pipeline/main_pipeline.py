#!/usr/bin/env python3
"""
Complete Footstep Audio Generation Pipeline
End-to-End: Video → Footstep Audio

Pipeline Flow:
1. Detect footsteps (timestamps + spatial data, includes video validation)
2. Analyze scene (surface, environment, footwear)
3. Generate audio variations via configurable backend
4. Prepare spatial data for audio processing
5. Process spatial audio (chop, pan, attenuate, mix)
6. Export final audio file (exact video duration)
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


from ..utils.logger import setup_logging, get_logger
logger = get_logger(__name__)

from .footstep_detector import FootstepDetector, DetectorConfig
from .scene_analyzer import SceneAnalyzer
from .spatial_audio_processor import SpatialAudioProcessor
from ..utils.video_merger import merge_audio_video, check_ffmpeg_installed
from ..utils.config import CAPTION_CONFIG_PATH, get_video_output_dir


try:
    from .audio_generator import generate_footsteps
    AUDIO_GENERATOR_AVAILABLE = True
except ImportError:
    # Audio generator not availble - using mock backend
    AUDIO_GENERATOR_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""

    # Audio generation
    backend: str = "runpod"  # Audio generation backend ("runpod", "mock", etc.)
    audio_variations: int = 1  # Number of variations to generate
    audio_length: float = 6.0  # Length of each variation (seconds)
    cfg_scale: float = 7.0
    steps: int = 100

    # Footstep detection
    target_fps: int = 10
    confidence_threshold: float = 0.7

    # Scene analysis
    scene_analysis_seed: int = 42

    # Spatial audio
    sample_rate: int = 44100
    fade_duration: float = 0.01
    min_attenuation_db: float = -20.0
    max_attenuation_db: float = 0.0

    # Output
    output_dir: str = "outputs/"
    save_intermediates: bool = True
    create_visualizations: bool = True
    merge_video: bool = True  # Merge audio with video (creates new file, default: True)

    def __post_init__(self):
        """Validate configuration"""
        # Import here to avoid circular dependency
        from ..audio_backends import list_backends

        available_backends = list_backends()
        if self.backend not in available_backends:
            raise ValueError(
                f"backend must be one of {available_backends}, got '{self.backend}'"
            )
        if self.audio_variations < 1:
            raise ValueError(f"audio_variations must be >= 1, got {self.audio_variations}")


class FootstepAudioPipeline:
    """Complete end-to-end pipeline for footstep audio generation"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with all components"""
        self.config = config if config is not None else PipelineConfig()
        
        print("=" * 80)
        print("FOOTSTEP AUDIO GENERATION PIPELINE")
        print("=" * 80)
        print(f"Audio Backend: {self.config.backend}")
        print(f"Audio variations: {self.config.audio_variations}")
        print(f"Output directory: {self.config.output_dir}")
        print()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing pipeline components...")
        # Note: VideoValidator is used internally by FootstepDetector
        self.footstep_detector = FootstepDetector(
            DetectorConfig(
                target_fps=self.config.target_fps,
                confidence_threshold=self.config.confidence_threshold
            )
        )
        # SceneAnalyzer requires caption_config.json path (use centralized config)
        self.scene_analyzer = SceneAnalyzer(
            caption_config_path=str(CAPTION_CONFIG_PATH),
            seed=self.config.scene_analysis_seed
        )
        self.spatial_processor = SpatialAudioProcessor(
            sample_rate=self.config.sample_rate,
            fade_duration=self.config.fade_duration,
            min_attenuation_db=self.config.min_attenuation_db,
            max_attenuation_db=self.config.max_attenuation_db
        )
        print("✓ All components initialized")
        print()
    
    def process_video(self, video_path: str, output_audio_path: Optional[str] = None) -> Dict:
        """
        Main pipeline: Video → Footstep Audio
        
        Args:
            video_path: Path to input video file
            output_audio_path: Optional custom output path for final audio
            
        Returns:
            Dictionary with processing results and metadata
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print("=" * 80)
        print(f"PROCESSING: {video_path.name}")
        print("=" * 80)
        print()

        # ====================================================================
        # STEP 1: Detect Footsteps (includes video validation)
        # ====================================================================
        print("STEP 1: Footstep Detection")
        print("-" * 80)
        detection_results = self.footstep_detector.process_video(str(video_path), verbose=True)

        # Get video info from detection results
        video_info = detection_results['video_info']
        
        num_footsteps = len(detection_results['heel_strike_detections'])
        if num_footsteps == 0:
            print("⚠️  No footsteps detected! Cannot generate audio.")
            return {'success': False, 'error': 'No footsteps detected'}
        
        print(f"\n✓ Detected {num_footsteps} footsteps")
        print()

        # ====================================================================
        # STEP 2: Scene Analysis
        # ====================================================================
        print("STEP 2: Scene Analysis")
        print("-" * 80)
        scene_results = self.scene_analyzer.analyze_from_detection_results(detection_results)

        audio_prompt = scene_results[0]['prompt']
        print(f"\n✓ Generated prompt: '{audio_prompt}'")
        print()

        # ====================================================================
        # STEP 3: Generate Audio Variations
        # ====================================================================
        print("STEP 3: Audio Generation")
        print("-" * 80)
        audio_variations = self._generate_audio_variations(audio_prompt)
        
        if not audio_variations:
            print("✗ Audio generation failed!")
            return {'success': False, 'error': 'Audio generation failed'}
        
        print(f"\n✓ Generated {len(audio_variations)} audio variations")
        print()

        # ====================================================================
        # STEP 4: Prepare Spatial Data
        # ====================================================================
        print("STEP 4: Preparing Spatial Audio Data")
        print("-" * 80)
        spatial_data = self._prepare_spatial_data(detection_results)
        print(f"✓ Prepared spatial data for {len(spatial_data)} footsteps")
        print()

        # ====================================================================
        # STEP 5: Process Spatial Audio
        # ====================================================================
        print("STEP 5: Spatial Audio Processing")
        print("-" * 80)
        
        # Determine output path
        if output_audio_path is None:
            output_audio_path = self.output_dir / f"{video_path.stem}_footsteps.wav"
        else:
            output_audio_path = Path(output_audio_path)
        
        final_audio = self._create_final_audio(
            audio_variations,
            spatial_data,
            video_info,
            output_audio_path
        )
        
        print(f"\n✓ Final audio saved: {output_audio_path}")
        print()

        # ====================================================================
        # OPTIONAL: Merge Audio with Video
        # ====================================================================
        merged_video_path = None
        if self.config.merge_video:
            print("OPTIONAL: Merging Audio with Video")
            print("-" * 80)

            if not check_ffmpeg_installed():
                print("⚠️  ffmpeg not installed - skipping video merge")
                print("   Install: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            else:
                try:
                    success, result = merge_audio_video(
                        video_path=str(video_path),
                        audio_path=str(output_audio_path),
                        output_path=None,  # Auto-generate name
                        output_dir=str(self.output_dir),  # Save to output directory
                        verbose=True
                    )
                    if success:
                        merged_video_path = result
                    else:
                        print(f"⚠️  Video merge failed: {result}")
                except Exception as e:
                    print(f"⚠️  Video merge error: {e}")

            print()

        # ====================================================================
        # STEP 6: Summary & Cleanup
        # ====================================================================
        total_time = time.time() - start_time
        
        print("=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Video duration: {video_info['duration']:.2f}s")
        print(f"Footsteps detected: {num_footsteps}")
        print(f"Audio variations generated: {len(audio_variations)}")
        print(f"Output audio: {output_audio_path}")
        if merged_video_path:
            print(f"Merged video: {merged_video_path}")
        print()
        
        # Compile results
        results = {
            'success': True,
            'video_path': str(video_path),
            'output_audio_path': str(output_audio_path),
            'merged_video_path': merged_video_path,  # None if not merged
            'processing_time_seconds': total_time,
            'video_info': video_info,
            'num_footsteps': num_footsteps,
            'num_audio_variations': len(audio_variations),
            'audio_prompt': audio_prompt,
            'scene_results': scene_results,
            'detection_results': detection_results
        }
        
        # Save metadata
        if self.config.save_intermediates:
            metadata_path = output_audio_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                # Create a copy and remove frames
                results_for_json = results.copy()
                if 'detection_results' in results_for_json and 'frames' in results_for_json['detection_results']:
                    # Remove frames to prevent huge JSON files and CPU spike
                    results_for_json['detection_results'] = results_for_json['detection_results'].copy()
                    del results_for_json['detection_results']['frames']

                # Convert non-serializable objects
                serializable_results = self._make_serializable(results_for_json)
                json.dump(serializable_results, f, indent=2)
            print(f"Metadata saved: {metadata_path}")

        # Cleanup resources to prevent hanging
        self.footstep_detector.cleanup()
        self.scene_analyzer.cleanup()

        return results

    def _generate_audio_variations(self, prompt: str) -> List[Tuple[np.ndarray, int, Path]]:
        """
        Step 4: Generate multiple audio variations
        
        Returns:
            List of (audio_array, sample_rate, file_path) tuples
        """
        if not AUDIO_GENERATOR_AVAILABLE:
            print("✗ Audio generator not available")
            return []
        
        variations = []
        variations_dir = self.output_dir / "variations"
        variations_dir.mkdir(exist_ok=True)

        print(f"Generating {self.config.audio_variations} variations...")
        print(f"Prompt: '{prompt}'")
        print(f"Backend: {self.config.backend}")
        print()

        for i in range(self.config.audio_variations):
            print(f"Variation {i+1}/{self.config.audio_variations}...")

            try:
                # Generate audio using audio_generator.py with pluggable backend
                audio_np, sr, audio_path, metadata = generate_footsteps(
                    prompt=prompt,
                    output_dir=str(variations_dir),
                    audio_length=self.config.audio_length,
                    cfg_scale=self.config.cfg_scale,
                    steps=self.config.steps,
                    backend=self.config.backend
                )
                
                variations.append((audio_np, sr, audio_path))
                
                print(f"  ✓ Generated: {audio_path.name}")
                print(f"  Time: {metadata['generation_time_seconds']}s")
                print()
                
            except Exception as e:
                print(f"  ✗ Failed to generate variation {i+1}: {e}")
                continue
        
        return variations
    
    def _prepare_spatial_data(self, detection_results: Dict) -> List[Dict]:
        """
        Step 5: Prepare spatial data for audio processing

        Extracts position and timing information from detection results.
        """
        # FootstepDetector already provides complete spatial_data
        if 'spatial_data' in detection_results and detection_results['spatial_data']:
            return detection_results['spatial_data']

        # Fallback: if no spatial data, create basic data from detections
        spatial_data = []
        for timestamp, foot_side in detection_results['heel_strike_detections']:
            spatial_info = {
                'timestamp': timestamp,
                'foot_side': foot_side,
                'x_position': 0.3 if foot_side == "LEFT" else 0.7,  # Simple L/R panning
                'hip_heel_pixel_distance': 200.0  # Default pixel distance
            }
            spatial_data.append(spatial_info)

        return spatial_data
    
    def _create_final_audio(
        self,
        audio_variations: List[Tuple[np.ndarray, int, str]],
        spatial_data: List[Dict],
        video_info: Dict,
        output_path: Path
    ) -> np.ndarray:
        """Use SpatialAudioProcessor to create final mix"""

        audio_array, sample_rate, _ = audio_variations[0]

        # Create detection_results format expected by SpatialAudioProcessor
        detection_results = {
            "video_info": video_info,
            "spatial_data": spatial_data
        }
        
        # Use SpatialAudioProcessor
        stats = self.spatial_processor.process_video_audio(
            audio_input=(audio_array, sample_rate),
            detection_results=detection_results,
            output_path=str(output_path),
            visualize=self.config.create_visualizations
        )
        
        return None  # Audio already saved by processor

    def _make_serializable(self, obj):
        """Convert non-JSON-serializable objects for metadata export"""
        from dataclasses import is_dataclass, asdict

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif is_dataclass(obj):
            return self._make_serializable(asdict(obj))  # Convert dataclass to dict
        else:
            return obj


def main():
    """Command-line interface for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete footstep audio generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Basic usage (RunPod backend)
  python -m src.main_pipeline video.mp4

  # Custom output path
  python -m src.main_pipeline video.mp4 --output footsteps.wav

  # Mock backend for testing
  python -m src.main_pipeline video.mp4 --backend mock 

  # Full customization
  python -m src.main_pipeline video.mp4 \\
    --output custom_output.wav \\
    --backend runpod \\
    --variations 10 \\
    --cfg-scale 8.0 \\
    --steps 120 \\
    --merge-video
        """
    )
    
    # Required arguments
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output audio file path (default: auto-generated in outputs/{video_name}_outputs/)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Audio generation backend: 'runpod' (default), 'mock' (testing), etc."
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=None,
        help="Number of audio variations to generate (default: 1)"
    )
    parser.add_argument(
        "--audio-length",
        type=float,
        default=None,
        help="Length of each audio variation in seconds (default: 6.0)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG scale for audio generation (default: 7.0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps for audio generation (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (default: outputs/{video_name}_outputs/)"
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Don't save intermediate files (variations, metadata)"
    )
    parser.add_argument(
        "--no-merge-video",
        action="store_true",
        help="Don't merge generated audio with original video (merge is default)"
    )

    args = parser.parse_args()

    # Set output directory based on video name if not specified
    if args.output_dir is None:
        args.output_dir = str(get_video_output_dir(args.video_path))

    # Create configuration - only pass CLI arguments
    config_kwargs = {}

    if args.backend is not None:
        config_kwargs['backend'] = args.backend
    if args.variations is not None:
        config_kwargs['audio_variations'] = args.variations
    if args.audio_length is not None:
        config_kwargs['audio_length'] = args.audio_length
    if args.cfg_scale is not None:
        config_kwargs['cfg_scale'] = args.cfg_scale
    if args.steps is not None:
        config_kwargs['steps'] = args.steps

    # Always set output_dir (already has default from get_video_output_dir)
    config_kwargs['output_dir'] = args.output_dir

    # Boolean flags (store_true) - always set based on CLI args
    config_kwargs['save_intermediates'] = not args.no_intermediates
    config_kwargs['merge_video'] = not args.no_merge_video

    config = PipelineConfig(**config_kwargs)
    
    # Create pipeline
    pipeline = FootstepAudioPipeline(config)
    
    try:
        # Process video
        results = pipeline.process_video(args.video_path, args.output)
        
        if results['success']:
            print("\n✅ SUCCESS!")
            print(f"Output audio: {results['output_audio_path']}")
            sys.exit(0)
        else:
            print(f"\n❌ FAILED: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()