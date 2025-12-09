import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the visualizers
from archive.pipeline_v1.full_footstep_detector import FootstepDetector, FootstepDetectorConfig
from src.utils.pose_extractor import PoseExtractor
from src.pipeline.video_validator import VideoValidator


class RawVsFullComparison:
    """
    Compare Raw MediaPipe vs Full Pipeline using A/B test visualization style
    
    Option B: Run both visualizers internally and compare results
    """
    
    def __init__(self, matching_tolerance: float = 0.3):
        self.matching_tolerance = matching_tolerance
        
        # Initialize components for raw MediaPipe
        self.pose_extractor = PoseExtractor(target_fps=10, confidence_threshold=0.7)
        self.video_validator = VideoValidator()
        
        # Raw MediaPipe parameters
        self.peak_height_threshold = 0.1
        self.peak_prominence = 0.05
        self.peak_distance = 5
    
    def run_raw_mediapipe_detection(self, video_path: str) -> Dict[str, Any]:
        """Run raw MediaPipe detection (Method A)"""
        print("Method A: Raw MediaPipe Detection")
        print("   Pipeline: MediaPipe -> Raw Distances -> Peak Detection -> Alternation")
        
        start_time = time.time()
        
        # Step 1: Extract pose landmarks
        landmarks_data, timestamps = self.pose_extractor.get_signal_processor_input(
            video_path, verbose=False
        )
        
        # Get video info
        video_info = self.video_validator.validate_video(video_path)
        
        # Step 2: Calculate raw hip-heel distances
        distance_data = self._calculate_raw_hip_heel_distances(landmarks_data, timestamps)
        
        # Step 3: Detect peaks
        left_peak_indices, left_peak_timestamps = self._detect_peaks_simple(
            distance_data['left_hip_heel'], distance_data['timestamps']
        )
        right_peak_indices, right_peak_timestamps = self._detect_peaks_simple(
            distance_data['right_hip_heel'], distance_data['timestamps']
        )
        
        # Step 4: Apply alternation filter
        filtered_detections = self._apply_alternation_filter(
            left_peak_timestamps, right_peak_timestamps
        )
        
        final_timestamps = [timestamp for timestamp, _ in filtered_detections]
        processing_time = time.time() - start_time
        
        return {
            'method': 'Raw MediaPipe (A)',
            'detected_timestamps': final_timestamps,
            'total_detections': len(final_timestamps),
            'processing_time': processing_time,
            'heel_strike_detections': filtered_detections,
            'distance_signals': {
                'left_hip_heel': distance_data['left_hip_heel'],
                'right_hip_heel': distance_data['right_hip_heel']
            },
            'timestamps': distance_data['timestamps'],
            'peak_details': {
                'left_peaks': left_peak_timestamps,
                'right_peaks': right_peak_timestamps,
                'left_peak_indices': left_peak_indices,
                'right_peak_indices': right_peak_indices
            },
            'video_info': video_info
        }
    
    def run_full_pipeline_detection(self, video_path: str) -> Dict[str, Any]:
        """Run full pipeline detection (Method B)"""
        print("Method B: Full Pipeline Detection")
        print("   Pipeline: MediaPipe -> Signal Processing -> Gait Detection -> Alternation")
        
        start_time = time.time()
        
        # Use the existing FootstepDetector
        detector = FootstepDetector()
        detection_results = detector.process_video(video_path, verbose=False)
        processing_time = time.time() - start_time
        
        # Clean up detector
        detector.cleanup()
        
        return {
            'method': 'Full Pipeline (B)',
            'detected_timestamps': detection_results['detected_timestamps'],
            'total_detections': detection_results['total_detections'],
            'processing_time': processing_time,
            'heel_strike_detections': detection_results.get('heel_strike_detections', []),
            'gait_results': detection_results['gait_results'],
            'processed_data': detection_results['processed_data'],
            'video_info': detection_results['video_info']
        }
    
    def load_ground_truth(self, ground_truth_path: str) -> Dict[str, Any]:
        """Load ground truth from JSON"""
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
        
        return {
            'timestamps': np.array([ann['timestamp'] for ann in data['annotations']]),
            'feet': [ann['foot'] for ann in data['annotations']],
            'total_steps': data['summary']['total_steps'],
            'left_steps': data['summary']['left_steps'],
            'right_steps': data['summary']['right_steps'],
            'video_info': data['video_info']
        }
    
    def calculate_accuracy_metrics(self, detected_timestamps: List[float], 
                                 ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics compared to ground truth"""
        gt_timestamps = ground_truth['timestamps']
        tolerance = self.matching_tolerance
        
        # Find matches
        matches = 0
        matched_gt_indices = set()
        
        for detected_time in detected_timestamps:
            for i, gt_time in enumerate(gt_timestamps):
                if i in matched_gt_indices:
                    continue
                if abs(detected_time - gt_time) <= tolerance:
                    matches += 1
                    matched_gt_indices.add(i)
                    break
        
        # Calculate metrics
        total_detected = len(detected_timestamps)
        total_ground_truth = len(gt_timestamps)
        
        detection_rate = (matches / total_ground_truth * 100) if total_ground_truth > 0 else 0
        precision = (matches / total_detected * 100) if total_detected > 0 else 0
        false_positives = total_detected - matches
        missed_detections = total_ground_truth - matches
        
        return {
            'detection_rate': detection_rate,
            'precision': precision,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'matches': matches,
            'total_detected': total_detected,
            'total_ground_truth': total_ground_truth
        }
    
    def create_comparison_visualization(self, raw_results: Dict, full_results: Dict, 
                                      ground_truth: Dict, video_name: str,
                                      save_path: str = None) -> None:
        """Create A/B test style comparison visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Raw MediaPipe vs Full Pipeline Comparison: {video_name}', 
                    fontsize=16, fontweight='bold')
        
        # Calculate accuracy metrics
        accuracy_raw = self.calculate_accuracy_metrics(raw_results['detected_timestamps'], ground_truth)
        accuracy_full = self.calculate_accuracy_metrics(full_results['detected_timestamps'], ground_truth)
        
        # 1. Detection Count Comparison (with ground truth)
        ax1 = axes[0, 0]
        methods = ['Raw MediaPipe\n(Method A)', 'Full Pipeline\n(Method B)']
        detection_counts = [raw_results['total_detections'], full_results['total_detections']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax1.bar(methods, detection_counts, color=colors, alpha=0.8)
        
        # Add ground truth line
        gt_count = ground_truth['total_steps']
        ax1.axhline(y=gt_count, color='gold', linestyle='--', linewidth=3, 
                   label=f'Ground Truth ({gt_count})', alpha=0.9)
        ax1.legend()
        
        ax1.set_title('Footsteps Detected vs Ground Truth', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Number of Detections', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars, detection_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 2. Accuracy Metrics
        ax2 = axes[0, 1]
        metrics = ['Detection Rate (%)', 'Precision (%)']
        raw_metrics = [accuracy_raw['detection_rate'], accuracy_raw['precision']]
        full_metrics = [accuracy_full['detection_rate'], accuracy_full['precision']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, raw_metrics, width, label='Raw MediaPipe', color='#FF6B6B', alpha=0.8)
        bars2 = ax2.bar(x + width/2, full_metrics, width, label='Full Pipeline', color='#4ECDC4', alpha=0.8)
        
        ax2.set_title('Accuracy Metrics vs Ground Truth', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bars_set in [bars1, bars2]:
            for bar in bars_set:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Processing Time Comparison
        ax3 = axes[1, 0]
        processing_times = [raw_results['processing_time'], full_results['processing_time']]
        
        bars3 = ax3.bar(methods, processing_times, color=colors, alpha=0.8)
        ax3.set_title('Processing Time', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Time (seconds)', fontsize=12)
        
        # Add value labels
        for bar, time_val in zip(bars3, processing_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Timeline Comparison
        ax4 = axes[1, 1]
        
        # Plot ground truth
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        for timestamp, foot in zip(gt_timestamps, gt_feet):
            color = 'blue' if foot == 'left' else 'red'
            ax4.scatter([timestamp], [2], c=color, s=80, marker='s', alpha=0.7, 
                       label='GT Left' if foot == 'left' and timestamp == gt_timestamps[0] 
                       else 'GT Right' if foot == 'right' and timestamp == gt_timestamps[0] 
                       else "")
        
        # Plot detected timestamps
        if raw_results['detected_timestamps']:
            ax4.scatter(raw_results['detected_timestamps'], 
                       [1] * len(raw_results['detected_timestamps']),
                       c='#FF6B6B', s=60, alpha=0.8, label='Raw MediaPipe')
        
        if full_results['detected_timestamps']:
            ax4.scatter(full_results['detected_timestamps'], 
                       [0.5] * len(full_results['detected_timestamps']),
                       c='#4ECDC4', s=60, alpha=0.8, label='Full Pipeline')
        
        ax4.set_title('Detection Timeline Comparison', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Time (seconds)', fontsize=12)
        ax4.set_ylabel('Method', fontsize=12)
        ax4.set_yticks([0.5, 1, 2])
        ax4.set_yticklabels(['Full Pipeline', 'Raw MediaPipe', 'Ground Truth'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error Analysis
        ax5 = axes[2, 0]
        error_types = ['False Positives', 'Missed Detections']
        raw_errors = [accuracy_raw['false_positives'], accuracy_raw['missed_detections']]
        full_errors = [accuracy_full['false_positives'], accuracy_full['missed_detections']]
        
        x = np.arange(len(error_types))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, raw_errors, width, label='Raw MediaPipe', color='#FF6B6B', alpha=0.8)
        bars2 = ax5.bar(x + width/2, full_errors, width, label='Full Pipeline', color='#4ECDC4', alpha=0.8)
        
        ax5.set_title('Error Analysis vs Ground Truth', fontweight='bold', fontsize=14)
        ax5.set_ylabel('Number of Errors', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(error_types)
        ax5.legend()
        
        # Add value labels
        for bars_set in [bars1, bars2]:
            for bar in bars_set:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 6. Summary Text
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Calculate differences and recommendation
        detection_diff = full_results['total_detections'] - raw_results['total_detections']
        time_diff = full_results['processing_time'] - raw_results['processing_time']
        time_overhead = (time_diff / raw_results['processing_time']) * 100 if raw_results['processing_time'] > 0 else 0
        accuracy_diff = accuracy_full['detection_rate'] - accuracy_raw['detection_rate']
        precision_diff = accuracy_full['precision'] - accuracy_raw['precision']
        
        # Determine recommendation
        if accuracy_diff > 10:
            recommendation = "✅ KEEP signal processing - much more accurate"
        elif accuracy_diff < -10:
            recommendation = "❌ REMOVE signal processing - raw method more accurate"
        elif abs(accuracy_diff) <= 10:
            if time_overhead > 50:
                recommendation = "🔥 REMOVE signal processing - similar accuracy, high cost"
            else:
                recommendation = "🤔 EITHER method works - similar accuracy"
        else:
            recommendation = "🔬 MIXED results - consider other factors"
        
        summary_text = f"""
RAW vs FULL PIPELINE COMPARISON
{'='*45}

📊 DETECTIONS:
   Raw MediaPipe (A): {raw_results['total_detections']} footsteps
   Full Pipeline (B): {full_results['total_detections']} footsteps
   Difference: {detection_diff:+d} footsteps

🎯 GROUND TRUTH COMPARISON:
   Total GT footsteps: {ground_truth['total_steps']}
   Raw accuracy: {accuracy_raw['detection_rate']:.1f}%
   Full accuracy: {accuracy_full['detection_rate']:.1f}%
   
   Raw precision: {accuracy_raw['precision']:.1f}%
   Full precision: {accuracy_full['precision']:.1f}%

⏱️ PROCESSING TIME:
   Raw: {raw_results['processing_time']:.3f}s
   Full: {full_results['processing_time']:.3f}s
   Overhead: {time_overhead:+.1f}%

💡 RECOMMENDATION:
{recommendation}

📝 SUMMARY:
Full pipeline is {accuracy_diff:+.1f}% more accurate
with {time_overhead:.1f}% time cost
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {save_path}")
    
    def compare_methods(self, video_name: str, 
                       test_videos_dir: str = "./test_videos",
                       output_dir: str = "./output/raw_vs_full_comparison") -> Dict[str, Any]:
        """
        Main comparison function - runs both methods and creates visualization
        
        Args:
            video_name: Name of video (e.g., "walk1")
            test_videos_dir: Directory containing test videos
            output_dir: Directory to save comparison plots
        """
        print("🆚 RAW MEDIAPIPE vs FULL PIPELINE COMPARISON")
        print("=" * 60)
        print(f"🎬 Analyzing: {video_name}")
        print()
        
        # Setup paths
        video_path = f"{test_videos_dir}/{video_name}.mp4"
        ground_truth_path = f"{test_videos_dir}/{video_name}_ground_truth.json"
        save_path = f"{output_dir}/{video_name}_raw_vs_full_comparison.png"
        
        try:
            # Step 1: Load ground truth
            print("📋 Loading ground truth...")
            ground_truth = self.load_ground_truth(ground_truth_path)
            print(f"   Ground truth: {ground_truth['total_steps']} steps")
            
            # Step 2: Run raw MediaPipe detection
            print("\n🔍 Running raw MediaPipe detection...")
            raw_results = self.run_raw_mediapipe_detection(video_path)
            print(f"   ✅ Raw detected {raw_results['total_detections']} footsteps")
            
            # Step 3: Run full pipeline detection
            print("\n🔄 Running full pipeline detection...")
            full_results = self.run_full_pipeline_detection(video_path)
            print(f"   ✅ Full detected {full_results['total_detections']} footsteps")
            
            # Step 4: Create comparison visualization
            print("\n📈 Creating comparison visualization...")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.create_comparison_visualization(
                raw_results, full_results, ground_truth, video_name, save_path
            )
            
            # Step 5: Print summary
            self._print_summary(raw_results, full_results, ground_truth)
            
            return {
                'video_name': video_name,
                'raw_results': raw_results,
                'full_results': full_results,
                'ground_truth': ground_truth,
                'save_path': save_path
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        finally:
            # Cleanup
            if self.pose_extractor:
                self.pose_extractor.cleanup()
    
    def _print_summary(self, raw_results: Dict, full_results: Dict, ground_truth: Dict):
        """Print detailed summary"""
        accuracy_raw = self.calculate_accuracy_metrics(raw_results['detected_timestamps'], ground_truth)
        accuracy_full = self.calculate_accuracy_metrics(full_results['detected_timestamps'], ground_truth)
        
        print("\n" + "🏆" + "=" * 58 + "🏆")
        print("                 COMPARISON RESULTS")
        print("🏆" + "=" * 58 + "🏆")
        
        print(f"\n📊 DETECTION RESULTS:")
        print(f"   Raw MediaPipe:      {raw_results['total_detections']} footsteps")
        print(f"   Full Pipeline:      {full_results['total_detections']} footsteps")
        print(f"   Ground Truth:       {ground_truth['total_steps']} footsteps")
        
        print(f"\n🎯 ACCURACY vs GROUND TRUTH:")
        print(f"   Raw Detection Rate:  {accuracy_raw['detection_rate']:.1f}%")
        print(f"   Full Detection Rate: {accuracy_full['detection_rate']:.1f}%")
        print(f"   Raw Precision:       {accuracy_raw['precision']:.1f}%")
        print(f"   Full Precision:      {accuracy_full['precision']:.1f}%")
        
        print(f"\n⏱️ PROCESSING TIME:")
        print(f"   Raw MediaPipe:       {raw_results['processing_time']:.3f} seconds")
        print(f"   Full Pipeline:       {full_results['processing_time']:.3f} seconds")
        
        # Recommendation
        accuracy_diff = accuracy_full['detection_rate'] - accuracy_raw['detection_rate']
        time_overhead = ((full_results['processing_time'] - raw_results['processing_time']) / 
                        raw_results['processing_time']) * 100
        
        print(f"\n💡 RECOMMENDATION:")
        if accuracy_diff > 10:
            print("   ✅ KEEP signal processing - significantly more accurate")
        elif accuracy_diff < -10:
            print("   ❌ REMOVE signal processing - raw method more accurate")
        elif abs(accuracy_diff) <= 10 and time_overhead > 50:
            print("   🔥 REMOVE signal processing - similar accuracy, high time cost")
        else:
            print("   🤔 EITHER method works - similar performance")
        
        print("=" * 60)
    
    # Helper methods for raw MediaPipe processing
    def _calculate_raw_hip_heel_distances(self, landmarks_data: List[Dict], 
                                        timestamps: List[float]) -> Dict[str, Any]:
        """Calculate hip-heel distances directly from raw MediaPipe data"""
        left_distances = []
        right_distances = []
        
        for landmarks, timestamp in zip(landmarks_data, timestamps):
            if landmarks is None:
                left_distances.append(np.nan)
                right_distances.append(np.nan)
                continue
            
            # Extract landmarks
            left_hip = landmarks.get('LEFT_HIP', (None, None, 0))
            left_heel = landmarks.get('LEFT_HEEL', (None, None, 0))
            right_hip = landmarks.get('RIGHT_HIP', (None, None, 0))
            right_heel = landmarks.get('RIGHT_HEEL', (None, None, 0))
            
            # Calculate left distance
            if (left_hip[0] is not None and left_heel[0] is not None and 
                left_hip[2] >= 0.7 and left_heel[2] >= 0.7):
                left_dist = np.sqrt((left_hip[0] - left_heel[0])**2 + (left_hip[1] - left_heel[1])**2)
                left_distances.append(left_dist)
            else:
                left_distances.append(np.nan)
            
            # Calculate right distance
            if (right_hip[0] is not None and right_heel[0] is not None and 
                right_hip[2] >= 0.7 and right_heel[2] >= 0.7):
                right_dist = np.sqrt((right_hip[0] - right_heel[0])**2 + (right_hip[1] - right_heel[1])**2)
                right_distances.append(right_dist)
            else:
                right_distances.append(np.nan)
        
        # Normalize distances
        def normalize_signal(signal):
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) < 2:
                return signal
            min_val = np.nanmin(signal)
            max_val = np.nanmax(signal)
            if max_val > min_val:
                return (signal - min_val) / (max_val - min_val)
            return signal
        
        left_distances = normalize_signal(np.array(left_distances))
        right_distances = normalize_signal(np.array(right_distances))
        
        return {
            'left_hip_heel': left_distances,
            'right_hip_heel': right_distances,
            'timestamps': np.array(timestamps),
            'data_coverage': np.sum(~np.isnan(left_distances)) / len(left_distances)
        }
    
    def _detect_peaks_simple(self, distances: np.ndarray, timestamps: np.ndarray) -> Tuple[List[int], List[float]]:
        """Simple peak detection on raw signals"""
        from scipy import signal
        
        valid_mask = ~np.isnan(distances)
        if np.sum(valid_mask) < 10:
            return [], []
        
        valid_distances = distances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        peak_indices_in_valid, _ = signal.find_peaks(
            valid_distances,
            height=self.peak_height_threshold,
            prominence=self.peak_prominence,
            distance=self.peak_distance
        )
        
        original_peak_indices = valid_indices[peak_indices_in_valid]
        peak_timestamps = timestamps[original_peak_indices]
        
        return original_peak_indices.tolist(), peak_timestamps.tolist()
    
    def _apply_alternation_filter(self, left_peaks: List[float], right_peaks: List[float]) -> List[Tuple[float, str]]:
        """Apply simple left-right alternation filter"""
        all_peaks = []
        for timestamp in left_peaks:
            all_peaks.append((timestamp, 'LEFT'))
        for timestamp in right_peaks:
            all_peaks.append((timestamp, 'RIGHT'))
        
        all_peaks.sort(key=lambda x: x[0])
        
        if not all_peaks:
            return []
        
        filtered = [all_peaks[0]]
        current_side = all_peaks[0][1]
        
        for timestamp, side in all_peaks[1:]:
            if side != current_side:
                filtered.append((timestamp, side))
                current_side = side
        
        return filtered


def main():
    """Example usage"""
    # Initialize comparator
    comparator = RawVsFullComparison(matching_tolerance=0.3)
    
    # Test with a single video
    VIDEO_NAME = "walk2"  # Change this to test different videos
    
    try:
        results = comparator.compare_methods(VIDEO_NAME)
        print(f"\n🎉 Comparison complete for {VIDEO_NAME}!")
        print(f"📁 Results saved to: {results['save_path']}")
        
    except FileNotFoundError:
        print(f"❌ Files not found for {VIDEO_NAME}")
        print("Available videos: walk1, walk2, walk3, walk4, walk5")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
