#!/usr/bin/env python3
"""
Scene Analyzer with CLIP-based Environment Detection
Generates footstep audio prompts from video scenes using environment classification.

Features:
- CLIP (ViT-B/32) for environment detection
- Caption generation matching training vocabulary
- Batch frame processing for stability
"""

import json
import random
import sys
import torch
import clip
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SceneSegment:
    """Represents a scene segment with timing and analysis results."""
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    environment: str
    confidence: float
    surface: str
    footwear: str
    prompt: str


# ============================================================================
# SCENE ANALYZER CLASS
# ============================================================================

class SceneAnalyzer:
    """
    Analyze video scenes and generate footstep audio prompts.
    
    Uses CLIP for environment classification and generates captions
    using the same vocabulary as training data.
    """
    
    def __init__(
        self,
        caption_config_path: str,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize scene analyzer.
        
        Args:
            caption_config_path: Path to caption_config.json
            device: Device to use ("cuda" or "cpu", auto-detect if None)
            seed: Random seed for reproducibility
        """
        print("=" * 80)
        print("Initializing Scene Analyzer")
        print("=" * 80)
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        # Device setup
        self.device = self._setup_device(device)
        
        # Load caption configuration
        self._load_caption_config(caption_config_path)
        
        # Initialize CLIP model
        self._initialize_clip()
        
        print("✓ Scene Analyzer ready")
        print("=" * 80)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device."""
        if device is not None:
            print(f"✓ Using specified device: {device}")
            return device
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU detected: {gpu_name}")
            return "cuda"
        else:
            print("⚠ No GPU detected, using CPU")
            return "cpu"
    
    def _load_caption_config(self, config_path: str) -> None:
        """Load caption configuration vocabulary."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Caption config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load vocabulary banks (same as training)
        self.surface_sound_map = config['surface_sound_map']
        self.footwear_materials = config['footwear_materials']
        self.timbres = config['timbres']
        self.intensities = config['intensities']
        self.locations = config['locations']
        self.actions = config['actions']
        
        # Define environment options for CLIP
        self.environment_options = config['environment_options']
        
        # Environment to surface mapping
        self.environment_surface_map = config['environment_surface_map']
        
        # Environment to footwear mapping
        self.environment_footwear_map = config['environment_footwear_map']
        
        print(f"✓ Loaded caption config: {len(self.surface_sound_map)} surfaces")
    
    def _initialize_clip(self) -> None:
        """Initialize CLIP model for environment classification."""
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print(f"✓ CLIP model loaded on {self.device}")
    
    # ========================================================================
    # MAIN ANALYSIS FUNCTIONS 
    # ========================================================================
    def analyze_video_segment(
        self,
        frames: List[np.ndarray],
        start_time: float,
        end_time: float,
        segment_id: int = 0,
        max_sample_frames: int = 5
    ) -> Dict:
        """
        Analyze a video segment and generate audio prompt.
        
        Args:
            frames: List of video frames (OpenCV BGR format)
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            segment_id: Identifier for this segment
            max_sample_frames: Max frames to sample for classification
        
        Returns:
            Dict with prompt and timing info for audio_generator
            Format: {
                "prompt": str,
                "seconds_start": 0.0,
                "seconds_total": 6.0
            }
        """
        if not frames:
            raise ValueError(f"Empty frame list for segment {segment_id}")
        
        duration = end_time - start_time
        
        print(f"\nAnalyzing segment {segment_id}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        
        # Sample frames for stable classification
        sample_frames = self._sample_frames(frames, max_sample_frames)
        
        # Classify environment using CLIP
        environment, confidence = self._classify_environment_batch(sample_frames)
        
        # Get contextual mappings
        surface = self._get_surface_for_environment(environment)
        footwear = self._get_footwear_for_environment(environment)
        
        # Generate comprehensive prompt (medium length)
        prompt = self._generate_caption(environment, surface, footwear)
        
        print(f"  Environment: {environment} (confidence: {confidence:.3f})")
        print(f"  Surface: {surface}, Footwear: {footwear}")
        print(f"  Prompt: '{prompt}'")
        
        # seconds_total is 6.0 (training length)
        return {
            "prompt": prompt,
            "seconds_start": 0.0,
            "seconds_total": 6.0
        }

    def analyze_from_detection_results(self, detection_results: Dict) -> List[Dict]:
        """
        Analyze scene from footstep detector results.

        This method extracts frames stored by FootstepDetector and performs scene analysis.

        Args:
            detection_results: Output from FootstepDetector.process_video() containing:
                - 'frames': List of raw frames (only frames with valid pose)
                - 'video_info': Video metadata dict

        Returns:
            List of scene analysis results (currently single segment for whole video):
            [
                {
                    "prompt": str,
                    "seconds_start": 0.0,
                    "seconds_total": 6.0
                }
            ]
        """
        # Extract frames from detection results
        frames = detection_results.get('frames', [])
        video_info = detection_results.get('video_info', {})

        if not frames:
            print("⚠️  No frames available for scene analysis, using default prompt")
            return [{
                'prompt': 'person walking on concrete floor with sneakers, high-quality',
                'environment': 'outdoor city street',
                'surface': 'concrete',
                'footwear': 'sneakers'
            }]

        # Get video duration
        video_duration = video_info.get('duration', 0.0)

        # Frames are already randomly sampled (50 frames from entire video)
        scene_result = self.analyze_video_segment(
            frames=frames,
            start_time=0.0,
            end_time=video_duration,
            segment_id=0,
            max_sample_frames=5  # Will sample 5 from the 50 stored frames
        )

        # Return as list for compatibility with pipeline
        return [scene_result]

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================
    
    def _sample_frames(
        self,
        frames: List[np.ndarray],
        max_frames: int
    ) -> List[np.ndarray]:
        """
        Sample representative frames from segment.
        
        Args:
            frames: All frames in segment
            max_frames: Maximum frames to sample
        
        Returns:
            Sampled frames evenly distributed across segment
        """
        if len(frames) <= max_frames:
            return frames
        
        # Evenly sample across segment
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    # ========================================================================
    # CLIP CLASSIFICATION
    # ========================================================================
    
    def _classify_environment_batch(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[str, float]:
        """
        Classify environment using batch of frames for stability.
        
        Uses majority voting across multiple frames to reduce
        single-frame classification errors.
        
        Args:
            frames: Sample frames from segment (BGR format)
        
        Returns:
            Tuple of (environment, confidence)
        """
        if not frames:
            return "outdoor city street", 0.5
        
        # Preprocess all frames
        processed_images = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            processed_image = self.preprocess(pil_image)
            processed_images.append(processed_image)
        
        # Batch process through CLIP
        image_batch = torch.stack(processed_images).to(self.device)
        text_tokens = clip.tokenize(self.environment_options).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.model(image_batch, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Majority vote: most frequent prediction
        predictions = np.argmax(probs, axis=1)
        prediction_counts = np.bincount(predictions, minlength=len(self.environment_options))
        final_prediction = np.argmax(prediction_counts)
        
        # Average confidence for winning prediction
        confidences_for_winner = probs[:, final_prediction]
        average_confidence = float(np.mean(confidences_for_winner))
        
        environment = self.environment_options[final_prediction]
        
        return environment, average_confidence
    
    # ========================================================================
    # CONTEXTUAL MAPPINGS
    # ========================================================================
    
    def _get_surface_for_environment(self, environment: str) -> str:
        """Map environment to surface type."""
        return self.environment_surface_map.get(environment, "concrete")
    
    def _get_footwear_for_environment(self, environment: str) -> str:
        """Map environment to appropriate footwear."""
        possible_footwear = self.environment_footwear_map.get(environment, ["sneakers"])
        return random.choice(possible_footwear)
    
    # ========================================================================
    # CAPTION GENERATION (MATCHING TRAINING VOCABULARY)
    # ========================================================================

    def _get_sound_descriptor(self, surface: str, footwear: str) -> str:
        """Get sound descriptor from surface_sound_map."""
        if surface not in self.surface_sound_map:
            return "stepping"
        
        if footwear not in self.surface_sound_map[surface]:
            # Fallback to first available footwear for this surface
            available = list(self.surface_sound_map[surface].keys())
            if available:
                footwear_alt = available[0]
                return random.choice(self.surface_sound_map[surface][footwear_alt])
            return "stepping"
        
        sounds = self.surface_sound_map[surface][footwear]
        return random.choice(sounds)
    
    def _generate_caption(
        self,
        environment: str,
        surface: str,
        footwear: str
    ) -> str:
        """
        Generate comprehensive medium-length prompt.
        
        Args:
            environment: Detected environment
            surface: Surface type
            footwear: Footwear type
        
        Returns:
            Comprehensive prompt string (medium length: 9-15 words)
        """
        # Get vocabulary elements
        material = random.choice(self.footwear_materials.get(footwear, [footwear.replace('_', ' ')]))
        location = random.choice(self.locations.get(surface, ['surface']))
        sound = self._get_sound_descriptor(surface, footwear)
        
        # Get character descriptors
        timbre = random.choice(self.timbres.get(footwear, ['steady']))
        intensity = random.choice(self.intensities.get(footwear, ['firm']))
        action = random.choice(self.actions)

        templates = [
            # Material + Character + Context (10-13 words)
            f"{material} {action} steadily on {surface} {location} producing {timbre} {sound}",
            f"{timbre} {intensity} {sound} from {material} on the {surface} {location}",
            f"{material} {action} rhythmically across {surface} {location} creating {timbre} {sound}",
            
            # Context + Material + Character (11-14 words)
            f"person wearing {material} {action} steadily across the {surface} {location}",
            f"{intensity} {timbre} {sound} from {material} making contact on {surface} {location}",
            
            # Material + Sound + Context (9-12 words)
            f"{material} {action} continuously on the {surface} {location}",
            f"{timbre} {sound} with {intensity} contact from {material} on {surface}",
            f"{material} {action} on {surface} {location} with distinct {sound}",
        ]
        
        caption = random.choice(templates)
        
        return caption
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def get_info(self) -> Dict:
        """Get analyzer configuration info."""
        return {
            "device": self.device,
            "num_environments": len(self.environment_options),
            "environments": self.environment_options,
            "num_surfaces": len(self.surface_sound_map),
            "surfaces": list(self.surface_sound_map.keys())
        }

    def cleanup(self):
        """Clean up CLIP model resources to prevent hanging."""
        if hasattr(self, 'model') and self.model is not None:
            # Delete model to free memory
            del self.model
            del self.preprocess

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = None
            self.preprocess = None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from ..utils.config import CAPTION_CONFIG_PATH, get_test_video, list_test_videos

    # Get configuration and test resources
    caption_config = str(CAPTION_CONFIG_PATH)

    # Get test video 
    try:
        test_video = str(get_test_video("your_test_video"))
    except FileNotFoundError:
        available = list_test_videos()
        if not available:
            print("ERROR: No test videos found in test_videos/ directory")
            sys.exit(1)
        test_video = str(get_test_video(available[0]))
        print(f"Using test video: {available[0]}")

    # Initialize analyzer
    analyzer = SceneAnalyzer(
        caption_config_path=caption_config,
        seed=42
    )
    
    # Load video
    cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {test_video}")
        sys.exit(1)
    
    # Extract frames (example: first 3 seconds)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(3 * fps)
    
    print(f"\nLoading video: {test_video}")
    print(f"FPS: {fps:.2f}")
    
    for frame_num in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        print("ERROR: Could not read frames from video")
        sys.exit(1)
    
    print(f"Loaded {len(frames)} frames")
    
    # Analyze single segment
    result = analyzer.analyze_video_segment(
        frames=frames,
        start_time=0.0,
        end_time=len(frames) / fps,
        segment_id=0
    )
    
    # Print result
    print("\n" + "=" * 80)
    print("ANALYSIS RESULT")
    print("=" * 80)
    print(f"Prompt: {result['prompt']}")
    print(f"Duration: {result['seconds_total']:.1f}s (always 6s for training)")
    print(f"Ready for audio_generator.generate_footsteps()")
    print("=" * 80)