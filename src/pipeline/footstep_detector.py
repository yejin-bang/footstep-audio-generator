import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

from ..utils.pose_extractor import PoseExtractor
from .video_validator import VideoValidator


@dataclass
class DetectorConfig:
    """Configuration for footstep detection"""
    # Pose extraction parameters
    target_fps: int = 10
    confidence_threshold: float = 0.7
    
    # Peak detection parameters
    peak_height_threshold: float = 0.1
    peak_prominence: float = 0.05
    peak_distance: int = 5  # frames
    
    def __post_init__(self):
        """Validate configuration"""
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


class FootstepDetector:

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector"""
        self.config = config if config is not None else DetectorConfig()

        # Initialize
        self.pose_extractor = PoseExtractor(
            target_fps=self.config.target_fps,
            confidence_threshold=self.config.confidence_threshold
        )
        self.video_validator = VideoValidator()
    
    def process_video(self, video_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            video_path: Path to video file
            verbose: Print processing information
            
        Returns:
            Dictionary containing detection results, spatial data, and metadata
        """
        if verbose:
            print("=" * 60)
            print("SIMPLE FOOTSTEP DETECTION WITH SPATIAL DATA")
            print("=" * 60)
            print("Architecture: Video -> Pose -> Distances -> Peaks -> Filter -> Spatial -> Output")
        
        # Step 1: Validate video
        if verbose:
            print("\nStep 1: Video validation...")
        video_info = self.video_validator.validate_video(video_path)

        # Step 2: Extract pose landmarks (pass video_info to avoid redundant validation)
        # Also store frames for scene analysis
        if verbose:
            print("\nStep 2: Extracting pose landmarks and storing frames...")
        pose_result = self.pose_extractor.process_video(
            video_path, video_info=video_info, verbose=verbose,
            store_frames=True, max_frames_to_store=50
        )
        landmarks_data = pose_result['landmarks_data']
        timestamps = pose_result['timestamps']
        stored_frames = pose_result.get('frames', [])
        
        # Step 3: Calculate hip-heel distances
        if verbose:
            print("\nStep 3: Calculating hip-heel distances...")
        distance_results = self._calculate_hip_heel_distances(landmarks_data, timestamps, verbose)
        
        # Step 4: Detect peaks on each foot
        if verbose:
            print("\nStep 4: Detecting peaks...")
        peak_results = self._detect_peaks(distance_results, verbose)
        
        # Step 5: Apply alternation filter
        if verbose:
            print("\nStep 5: Applying alternation filter...")
        final_detections = self._apply_alternation_filter(peak_results, verbose)
        
        # Step 6: Extract spatial data at detection timestamps
        if verbose:
            print("\nStep 6: Extracting spatial data...")
        spatial_data = self._extract_spatial_data_at_detections(
            final_detections,
            landmarks_data,
            distance_results['timestamps'],  # Use numpy array, not list
            verbose
        )
        
        # Step 7: Compile results
        detection_results = {
            # Main outputs for audio processing
            'detected_timestamps': [timestamp for timestamp, _ in final_detections],
            'heel_strike_detections': final_detections,  # (timestamp, foot_side) pairs
            'total_detections': len(final_detections),

            # Spatial data for audio spatialization
            'spatial_data': spatial_data,  # List of dicts with x_position, hip_heel_distance, etc.

            # Stored frames for scene analysis (only frames with valid pose)
            'frames': stored_frames,  # Raw frames for scene_analyzer

            # Detailed data for analysis
            'distance_signals': {
                'left_hip_heel': distance_results['left_distances'],
                'right_hip_heel': distance_results['right_distances'],
                'timestamps': distance_results['timestamps']
            },
            'peak_details': peak_results,

            # Metadata
            'video_info': video_info,
            'processing_config': self.config,
            'processing_method': 'simple_direct_with_spatial',
            'data_coverage': distance_results['data_coverage']
        }
        
        if verbose:
            print(f"\nDetection complete!")
            print(f"   Total detections: {len(final_detections)}")
            print(f"   Spatial data extracted: {len(spatial_data)}/{len(final_detections)}")
            if final_detections:
                print(f"\n   Final gait pattern with spatial info:")
                for i, spatial_info in enumerate(spatial_data[:5]):  # Show first 5
                    print(f"     {i+1:2d}. {spatial_info['timestamp']:.3f}s - "
                          f"{spatial_info['foot_side']:5s} foot - "
                          f"x:{spatial_info['x_position']:.3f} - "
                          f"dist:{spatial_info['hip_heel_pixel_distance']:.3f}")
                if len(spatial_data) > 5:
                    print(f"     ... and {len(spatial_data)-5} more")
        
        return detection_results
    
    def _calculate_hip_heel_distances(self, landmarks_data: List[Dict], 
                                    timestamps: List[float], 
                                    verbose: bool = False) -> Dict[str, Any]:
        """Calculate and normalize hip-heel distances (from raw visualizer)"""
        
        left_distances = []
        right_distances = []
        valid_frame_count = 0
        
        for landmarks in landmarks_data:
            if landmarks is None:
                left_distances.append(np.nan)
                right_distances.append(np.nan)
                continue
            
            # Extract landmark coordinates
            left_hip = landmarks.get('LEFT_HIP', (None, None, 0))
            left_heel = landmarks.get('LEFT_HEEL', (None, None, 0))
            right_hip = landmarks.get('RIGHT_HIP', (None, None, 0))
            right_heel = landmarks.get('RIGHT_HEEL', (None, None, 0))
            
            frame_has_data = False
            
            # Calculate left hip-heel distance
            if (left_hip[0] is not None and left_heel[0] is not None and 
                left_hip[2] >= self.config.confidence_threshold and 
                left_heel[2] >= self.config.confidence_threshold):
                left_dist = np.sqrt((left_hip[0] - left_heel[0])**2 + (left_hip[1] - left_heel[1])**2)
                left_distances.append(left_dist)
                frame_has_data = True
            else:
                left_distances.append(np.nan)
            
            # Calculate right hip-heel distance
            if (right_hip[0] is not None and right_heel[0] is not None and 
                right_hip[2] >= self.config.confidence_threshold and 
                right_heel[2] >= self.config.confidence_threshold):
                right_dist = np.sqrt((right_hip[0] - right_heel[0])**2 + (right_hip[1] - right_heel[1])**2)
                right_distances.append(right_dist)
                frame_has_data = True
            else:
                right_distances.append(np.nan)
            
            if frame_has_data:
                valid_frame_count += 1
        
        # Convert to numpy arrays
        left_distances = np.array(left_distances)
        right_distances = np.array(right_distances)
        timestamps_array = np.array(timestamps)
        
        # Normalize distances (0-1 range)
        left_distances = self._normalize_signal(left_distances)
        right_distances = self._normalize_signal(right_distances)
        
        data_coverage = valid_frame_count / len(landmarks_data) if landmarks_data else 0
        
        if verbose:
            print(f"   Data coverage: {data_coverage:.1%}")
            print(f"   Left distance range: {np.nanmin(left_distances):.3f} to {np.nanmax(left_distances):.3f}")
            print(f"   Right distance range: {np.nanmin(right_distances):.3f} to {np.nanmax(right_distances):.3f}")
        
        return {
            'left_distances': left_distances,
            'right_distances': right_distances,
            'timestamps': timestamps_array,
            'data_coverage': data_coverage
        }
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to 0-1 range"""
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) < 2:
            return signal
        
        min_val = np.nanmin(signal)
        max_val = np.nanmax(signal)
        if max_val > min_val:
            normalized = (signal - min_val) / (max_val - min_val)
            return normalized
        return signal
    
    def _detect_peaks(self, distance_results: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Detect peaks in distance signals"""
        
        left_distances = distance_results['left_distances']
        right_distances = distance_results['right_distances']
        timestamps = distance_results['timestamps']
        
        # Detect left peaks
        left_peak_indices, left_peak_timestamps = self._find_peaks_in_signal(
            left_distances, timestamps, "left"
        )
        
        # Detect right peaks
        right_peak_indices, right_peak_timestamps = self._find_peaks_in_signal(
            right_distances, timestamps, "right"
        )
        
        if verbose:
            print(f"   Left peaks found: {len(left_peak_timestamps)}")
            print(f"   Right peaks found: {len(right_peak_timestamps)}")
            print(f"   Total raw peaks: {len(left_peak_timestamps) + len(right_peak_timestamps)}")
        
        return {
            'left_peaks': left_peak_timestamps,
            'right_peaks': right_peak_timestamps,
            'left_peak_indices': left_peak_indices,
            'right_peak_indices': right_peak_indices
        }
    
    def _find_peaks_in_signal(self, distances: np.ndarray, timestamps: np.ndarray, 
                            side: str) -> Tuple[List[int], List[float]]:
        """Find peaks in a single distance signal"""
        
        # Remove NaN values for peak detection
        valid_mask = ~np.isnan(distances)
        if np.sum(valid_mask) < 10:  # Need at least 10 valid points
            return [], []
        
        valid_distances = distances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Find peaks using scipy
        peak_indices_in_valid, _ = signal.find_peaks(
            valid_distances,
            height=self.config.peak_height_threshold,
            prominence=self.config.peak_prominence,
            distance=self.config.peak_distance
        )
        
        # Convert back to original indices and timestamps
        if len(peak_indices_in_valid) > 0:
            original_peak_indices = valid_indices[peak_indices_in_valid]
            peak_timestamps = timestamps[original_peak_indices]
            return original_peak_indices.tolist(), peak_timestamps.tolist()
        else:
            return [], []
    
    def _apply_alternation_filter(self, peak_results: Dict[str, Any], 
                                verbose: bool = False) -> List[Tuple[float, str]]:
        """Apply left-right alternation filter"""
        
        left_peaks = peak_results['left_peaks']
        right_peaks = peak_results['right_peaks']
        
        # Combine and sort all peaks
        all_peaks = []
        for timestamp in left_peaks:
            all_peaks.append((timestamp, 'LEFT'))
        for timestamp in right_peaks:
            all_peaks.append((timestamp, 'RIGHT'))
        
        all_peaks.sort(key=lambda x: x[0])
        
        if not all_peaks:
            if verbose:
                print("   No peaks to filter")
            return []
        
        # Apply pure alternation logic
        filtered = [all_peaks[0]]  # Always keep first detection
        current_side = all_peaks[0][1]
        
        for timestamp, side in all_peaks[1:]:
            if side != current_side:
                filtered.append((timestamp, side))
                current_side = side
            # Skip same-side detections
        
        if verbose:
            raw_count = len(all_peaks)
            filtered_count = len(filtered)
            removed_count = raw_count - filtered_count
            print(f"   Raw peaks: {raw_count}")
            print(f"   After alternation filter: {filtered_count}")
            print(f"   Same-side repetitions removed: {removed_count}")
        
        return filtered
    
    def _extract_spatial_data_at_detections(self, detection_timestamps: List[Tuple[float, str]], 
                                           landmarks_data: List[Dict],
                                           timestamps_array: np.ndarray,
                                           verbose: bool = False) -> List[Dict]:
        """
        Extract spatial information (x-position, hip-heel pixel distance) at each detected footstep
        
        This data is used for audio spatialization:
        - x_position: for stereo panning (left/right positioning)
        - hip_heel_pixel_distance: for volume attenuation (depth/distance from camera)
        
        Args:
            detection_timestamps: List of (timestamp, foot_side) tuples
            landmarks_data: Raw landmark data from MediaPipe
            timestamps_array: Array of frame timestamps
            verbose: Print extraction info
            
        Returns:
            List of dictionaries with spatial data for each detection
        """
        spatial_data = []
        
        for det_timestamp, foot_side in detection_timestamps:
            # Find closest frame to this timestamp
            time_diffs = np.abs(timestamps_array - det_timestamp)
            closest_frame_idx = np.argmin(time_diffs)
            
            landmarks = landmarks_data[closest_frame_idx]
            
            if landmarks is None:
                if verbose:
                    print(f"   Warning: No landmarks at {det_timestamp:.3f}s")
                continue
            
            # Extract landmark coordinates
            left_hip = landmarks.get('LEFT_HIP', (None, None, 0))
            right_hip = landmarks.get('RIGHT_HIP', (None, None, 0))
            left_heel = landmarks.get('LEFT_HEEL', (None, None, 0))
            right_heel = landmarks.get('RIGHT_HEEL', (None, None, 0))
            
            # Calculate center position (average of both hips) for panning
            # This represents the horizontal position of the person in frame
            # Fallback: use single hip if both aren't visible (common in side-angle shots)
            if (left_hip[0] is not None and right_hip[0] is not None and
                left_hip[2] >= self.config.confidence_threshold and
                right_hip[2] >= self.config.confidence_threshold):
                # Best case: average both hips
                center_x = (left_hip[0] + right_hip[0]) / 2.0
            elif (left_hip[0] is not None and left_hip[2] >= self.config.confidence_threshold):
                # Fallback: use left hip only
                center_x = left_hip[0]
            elif (right_hip[0] is not None and right_hip[2] >= self.config.confidence_threshold):
                # Fallback: use right hip only
                center_x = right_hip[0]
            else:
                center_x = None
            
            # Calculate hip-heel pixel distance for the striking foot
            if foot_side == 'LEFT':
                if (left_hip[0] is not None and left_heel[0] is not None and
                    left_hip[2] >= self.config.confidence_threshold and
                    left_heel[2] >= self.config.confidence_threshold):
                    # Pixel distance (NOT normalized - this preserves apparent size)
                    hip_heel_distance = np.sqrt(
                        (left_hip[0] - left_heel[0])**2 + 
                        (left_hip[1] - left_heel[1])**2
                    )
                    confidence = (left_hip[2] + left_heel[2]) / 2.0
                else:
                    hip_heel_distance = None
                    confidence = 0.0
            else:  # RIGHT foot
                if (right_hip[0] is not None and right_heel[0] is not None and
                    right_hip[2] >= self.config.confidence_threshold and
                    right_heel[2] >= self.config.confidence_threshold):
                    hip_heel_distance = np.sqrt(
                        (right_hip[0] - right_heel[0])**2 + 
                        (right_hip[1] - right_heel[1])**2
                    )
                    confidence = (right_hip[2] + right_heel[2]) / 2.0
                else:
                    hip_heel_distance = None
                    confidence = 0.0
            
            spatial_info = {
                'timestamp': det_timestamp,
                'foot_side': foot_side,
                'x_position': center_x,  # Raw pixel coordinate
                'hip_heel_pixel_distance': hip_heel_distance,  # Unnormalized pixel distance
                'confidence': confidence
            }
            
            spatial_data.append(spatial_info)
        
        if verbose:
            print(f"   Extracted spatial data for {len(spatial_data)}/{len(detection_timestamps)} detections")
            if spatial_data:
                valid_distances = [d['hip_heel_pixel_distance'] for d in spatial_data 
                                 if d['hip_heel_pixel_distance'] is not None]
                if valid_distances:
                    print(f"   Hip-heel distance range: {min(valid_distances):.3f} to {max(valid_distances):.3f} pixels")
                    print(f"   Reference (max) distance: {max(valid_distances):.3f} pixels (closest point)")
        
        return spatial_data
    
    def get_footstep_timestamps(self, video_path: str, verbose: bool = False) -> List[float]:
        """
        Convenience method to get just the timestamps
        
        Args:
            video_path: Path to video file
            verbose: Print processing information
            
        Returns:
            List of footstep timestamps in seconds
        """
        results = self.process_video(video_path, verbose=verbose)
        return results['detected_timestamps']
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_extractor:
            self.pose_extractor.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Test the detector with spatial data
    config = DetectorConfig(
        target_fps=10,
        confidence_threshold=0.7,
        peak_height_threshold=0.1,
        peak_prominence=0.05,
        peak_distance=5
    )
    
    print("Simple Footstep Detector with Spatial Data Extraction")
    print("=" * 60)
    print("Architecture: Single-class, direct processing + spatial extraction")
    print("Based on: Proven raw visualizer logic")
    print(f"Configuration: {config}")
    
    # Initialize detector
    detector = FootstepDetector(config)
    
    # Test video path
    video_path = "path_to_your_video"
    
    try:
        # Process video
        results = detector.process_video(video_path, verbose=True)
        
        print(f"\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"   Detected {results['total_detections']} footsteps")
        print(f"   Data coverage: {results['data_coverage']:.1%}")
        print(f"   Processing method: {results['processing_method']}")
        
        # Show detections with spatial data
        if results['spatial_data']:
            print(f"\n   Detailed spatial data:")
            for i, spatial_info in enumerate(results['spatial_data']):
                print(f"     {i+1:2d}. t={spatial_info['timestamp']:.3f}s | "
                      f"{spatial_info['foot_side']:5s} | "
                      f"x={spatial_info['x_position']:.3f} | "
                      f"depth={spatial_info['hip_heel_pixel_distance']:.3f}px | "
                      f"conf={spatial_info['confidence']:.2f}")
        
        print(f"\n   Ready for spatial audio processing!")
        
    except FileNotFoundError:
        print(f"Video not found: {video_path}")
        print("Please update the video path")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        detector.cleanup()
        print("\nDetector cleanup complete!")