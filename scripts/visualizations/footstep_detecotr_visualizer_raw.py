import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.pose_extractor import PoseExtractor
from src.pipeline.video_validator import VideoValidator


@dataclass
class RawVisualizerConfig:
    """Configuration for raw MediaPipe visualization"""
    matching_tolerance: float = 0.3  # ±0.3 seconds for ground truth matching
    figure_size: Tuple[int, int] = (20, 16)  # Large size for combined view
    dpi: int = 100


class RawMediaPipeFootstepDetector:
    """
    Direct MediaPipe to footstep detection - skipping signal processor
    
    Pipeline:
    MediaPipe -> Raw Hip-Heel distances -> Peak detection -> Visualization
    """
    
    def __init__(self, target_fps: int = 10, confidence_threshold: float = 0.7):
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.pose_extractor = PoseExtractor(
            target_fps=target_fps, 
            confidence_threshold=confidence_threshold
        )
        self.video_validator = VideoValidator()
        
        # Peak detection parameters
        self.peak_height_threshold = 0.1
        self.peak_prominence = 0.05
        self.peak_distance = 5  # frames
    
    def calculate_raw_hip_heel_distances(self, landmarks_data: List[Dict], 
                                       timestamps: List[float]) -> Dict[str, Any]:
        """Calculate hip-heel distances directly from raw MediaPipe data"""
        
        left_distances = []
        right_distances = []
        valid_timestamps = []
        
        for i, (landmarks, timestamp) in enumerate(zip(landmarks_data, timestamps)):
            if landmarks is None:
                left_distances.append(np.nan)
                right_distances.append(np.nan)
                valid_timestamps.append(timestamp)
                continue
            
            # Extract landmarks
            left_hip = landmarks.get('LEFT_HIP', (None, None, 0))
            left_heel = landmarks.get('LEFT_HEEL', (None, None, 0))
            right_hip = landmarks.get('RIGHT_HIP', (None, None, 0))
            right_heel = landmarks.get('RIGHT_HEEL', (None, None, 0))
            
            # Calculate left distance
            if (left_hip[0] is not None and left_heel[0] is not None and 
                left_hip[2] >= self.confidence_threshold and left_heel[2] >= self.confidence_threshold):
                left_dist = np.sqrt((left_hip[0] - left_heel[0])**2 + (left_hip[1] - left_heel[1])**2)
                left_distances.append(left_dist)
                
            else:
                left_distances.append(np.nan)
            
            # Calculate right distance
            if (right_hip[0] is not None and right_heel[0] is not None and 
                right_hip[2] >= self.confidence_threshold and right_heel[2] >= self.confidence_threshold):
                right_dist = np.sqrt((right_hip[0] - right_heel[0])**2 + (right_hip[1] - right_heel[1])**2)
                right_distances.append(right_dist)
            else:
                right_distances.append(np.nan)
            
            valid_timestamps.append(timestamp)
        
        # Convert to numpy arrays
        left_distances = np.array(left_distances)
        right_distances = np.array(right_distances)
        valid_timestamps = np.array(valid_timestamps)
        
        # Simple normalization (0-1 range)
        def normalize_signal(signal):
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) < 2:
                return signal
            
            min_val = np.nanmin(signal)
            max_val = np.nanmax(signal)
            if max_val > min_val:
                normalized = (signal - min_val) / (max_val - min_val)
                return normalized
            return signal
        
        left_distances = normalize_signal(left_distances)
        right_distances = normalize_signal(right_distances)
        
        return {
            'left_hip_heel': left_distances,
            'right_hip_heel': right_distances,
            'timestamps': valid_timestamps,
            'data_coverage': np.sum(~np.isnan(left_distances)) / len(left_distances)
        }
    
    def detect_peaks_simple(self, distances: np.ndarray, timestamps: np.ndarray) -> Tuple[List[int], List[float]]:
        """Simple peak detection on raw signals"""
        # Remove NaN values for peak detection
        valid_mask = ~np.isnan(distances)
        if np.sum(valid_mask) < 10:  # Need at least 10 valid points
            return [], []
        
        valid_distances = distances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        valid_timestamps = timestamps[valid_mask]
        
        # Find peaks
        peak_indices_in_valid, _ = signal.find_peaks(
            valid_distances,
            height=self.peak_height_threshold,
            prominence=self.peak_prominence,
            distance=self.peak_distance
        )
        
        # Convert back to original indices
        original_peak_indices = valid_indices[peak_indices_in_valid]
        peak_timestamps = timestamps[original_peak_indices]
        
        return original_peak_indices.tolist(), peak_timestamps.tolist()
    
    def apply_alternation_filter(self, left_peaks: List[float], right_peaks: List[float]) -> List[Tuple[float, str]]:
        """Apply simple left-right alternation filter"""
        # Combine and sort all peaks
        all_peaks = []
        for timestamp in left_peaks:
            all_peaks.append((timestamp, 'LEFT'))
        for timestamp in right_peaks:
            all_peaks.append((timestamp, 'RIGHT'))
        
        all_peaks.sort(key=lambda x: x[0])
        
        if not all_peaks:
            return []
        
        # Apply alternation filter
        filtered = [all_peaks[0]]
        current_side = all_peaks[0][1]
        
        for timestamp, side in all_peaks[1:]:
            if side != current_side:
                filtered.append((timestamp, side))
                current_side = side
        
        return filtered
    
    def process_video(self, video_path: str, verbose: bool = True) -> Dict[str, Any]:
        """Process video with raw MediaPipe approach"""
        if verbose:
            print("=" * 60)
            print("RAW MEDIAPIPE FOOTSTEP DETECTION")
            print("=" * 60)
        
        # Step 1: Extract pose landmarks
        if verbose:
            print("\n🏃 Step 1: Extracting pose landmarks...")
        
        landmarks_data, timestamps = self.pose_extractor.get_signal_processor_input(
            video_path, verbose=verbose
        )
        
        # Get video info
        video_info = self.video_validator.validate_video(video_path)
        
        # Step 2: Calculate raw hip-heel distances
        if verbose:
            print("\n🔍 Step 2: Calculating raw hip-heel distances...")
        
        distance_data = self.calculate_raw_hip_heel_distances(landmarks_data, timestamps)
        
        if verbose:
            print(f"   Data coverage: {distance_data['data_coverage']:.1%}")
        
        # Step 3: Detect peaks
        if verbose:
            print("\n🎯 Step 3: Detecting peaks...")
        
        left_peak_indices, left_peak_timestamps = self.detect_peaks_simple(
            distance_data['left_hip_heel'], distance_data['timestamps']
        )
        right_peak_indices, right_peak_timestamps = self.detect_peaks_simple(
            distance_data['right_hip_heel'], distance_data['timestamps']
        )
        
        if verbose:
            print(f"   Left peaks: {len(left_peak_timestamps)}")
            print(f"   Right peaks: {len(right_peak_timestamps)}")
        
        # Step 4: Apply alternation filter
        if verbose:
            print("\n🔄 Step 4: Applying alternation filter...")
        
        filtered_detections = self.apply_alternation_filter(
            left_peak_timestamps, right_peak_timestamps
        )
        
        final_timestamps = [timestamp for timestamp, _ in filtered_detections]
        
        if verbose:
            print(f"   Raw total: {len(left_peak_timestamps) + len(right_peak_timestamps)}")
            print(f"   After alternation: {len(filtered_detections)}")
            
            if filtered_detections:
                print("\n🚶 Final pattern:")
                for i, (timestamp, foot) in enumerate(filtered_detections):
                    print(f"   {i+1:2d}. {timestamp:.3f}s - {foot} foot")
        
        return {
            'detected_timestamps': final_timestamps,
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
            'video_info': video_info,
            'processing_method': 'raw_mediapipe',
            'processed_fps': self.target_fps,  # Add this for filename
            'total_detections': len(filtered_detections)
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_extractor:
            self.pose_extractor.cleanup()


class RawMediaPipeVisualizer:
    """
    NEW: Combined visualizer for raw MediaPipe results
    Same 2x2 layout as the main visualizer
    """
    
    def __init__(self, config: Optional[RawVisualizerConfig] = None):
        self.config = config if config is not None else RawVisualizerConfig()
    
    def load_ground_truth(self, json_path: str) -> Dict[str, Any]:
        """Load ground truth from JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            'timestamps': np.array([ann['timestamp'] for ann in data['annotations']]),
            'feet': [ann['foot'] for ann in data['annotations']],
            'frames': np.array([ann['frame'] for ann in data['annotations']]),
            'total_steps': data['summary']['total_steps'],
            'left_steps': data['summary']['left_steps'],
            'right_steps': data['summary']['right_steps'],
            'video_info': data['video_info'],
            'video_path': data['video_path']
        }
    
    def compare_with_ground_truth(self, detected_timestamps: List[float], 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare detected vs ground truth"""
        gt_timestamps = ground_truth['timestamps']
        tolerance = self.config.matching_tolerance
        
        matches = []
        matched_gt_indices = set()
        matched_detected_indices = set()
        
        for i, detected_time in enumerate(detected_timestamps):
            for j, gt_time in enumerate(gt_timestamps):
                if j in matched_gt_indices:
                    continue
                
                time_diff = abs(detected_time - gt_time)
                if time_diff <= tolerance:
                    matches.append({
                        'detected_index': i,
                        'detected_timestamp': detected_time,
                        'gt_index': j,
                        'gt_timestamp': gt_time,
                        'time_error': detected_time - gt_time,
                        'abs_time_error': time_diff,
                        'gt_foot': ground_truth['feet'][j],
                        'gt_frame': ground_truth['frames'][j]
                    })
                    matched_gt_indices.add(j)
                    matched_detected_indices.add(i)
                    break
        
        # Calculate metrics
        metrics = self.calculate_metrics(matches, detected_timestamps, ground_truth)
        
        # Identify false positives and missed detections
        false_positives = [
            {'index': i, 'timestamp': ts} 
            for i, ts in enumerate(detected_timestamps) 
            if i not in matched_detected_indices
        ]
        
        missed_detections = [
            {'index': j, 'timestamp': gt_timestamps[j], 'foot': ground_truth['feet'][j]}
            for j in range(len(gt_timestamps))
            if j not in matched_gt_indices
        ]
        
        return {
            'matches': matches,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'metrics': metrics,
            'matching_tolerance': tolerance,
            'total_detected': len(detected_timestamps),
            'total_ground_truth': len(gt_timestamps)
        }
    
    def calculate_metrics(self, matches: List[Dict], 
                         detected_timestamps: List[float],
                         ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from matching results"""
        n_matches = len(matches)
        n_detected = len(detected_timestamps)
        n_ground_truth = len(ground_truth['timestamps'])
        
        # Basic metrics
        detection_rate = (n_matches / n_ground_truth * 100) if n_ground_truth > 0 else 0
        precision = (n_matches / n_detected * 100) if n_detected > 0 else 0
        false_positive_rate = ((n_detected - n_matches) / n_detected * 100) if n_detected > 0 else 0
        
        # Time accuracy metrics
        if n_matches > 0:
            time_errors = [match['time_error'] for match in matches]
            abs_time_errors = [match['abs_time_error'] for match in matches]
            
            mean_time_error = np.mean(time_errors)
            std_time_error = np.std(time_errors)
            mean_abs_error = np.mean(abs_time_errors)
            max_abs_error = np.max(abs_time_errors)
        else:
            mean_time_error = 0
            std_time_error = 0
            mean_abs_error = 0
            max_abs_error = 0
        
        # F1 Score
        if precision + detection_rate > 0:
            f1_score = 2 * (precision * detection_rate / 100) / (precision + detection_rate) * 100
        else:
            f1_score = 0
        
        return {
            'detection_rate': detection_rate,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'f1_score': f1_score,
            'mean_time_error': mean_time_error,
            'std_time_error': std_time_error,
            'mean_abs_error': mean_abs_error,
            'max_abs_error': max_abs_error,
            'total_matches': n_matches,
            'total_detected': n_detected,
            'total_ground_truth': n_ground_truth
        }

    def create_combined_analysis_plot(self, detection_results: Dict[str, Any], 
                                    ground_truth: Dict[str, Any],
                                    comparison_results: Dict[str, Any],
                                    save_path: str = None) -> None:
        """
        NEW: Create comprehensive combined plot - same as main visualizer
        
        Layout:
        +---------------------------+---------------------------+
        |  Raw Distance Signals     |    Performance Metrics    |
        |  (with GT & peaks)        |    (bars + text summary)  |
        +---------------------------+---------------------------+
        |  Timeline Comparison      |    Error Analysis         |
        |  (GT vs Detected)         |    (detailed breakdown)   |
        +---------------------------+---------------------------+
        """
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size, 
                                                     dpi=self.config.dpi)
        
        # Get data
        distance_signals = detection_results['distance_signals']
        peak_details = detection_results['peak_details']
        timestamps = detection_results['timestamps']
        detected_timestamps = detection_results['detected_timestamps']
        metrics = comparison_results['metrics']
        
        # SUBPLOT 1: Raw Distance Signals (Top Left)
        self._plot_raw_distance_signals(ax1, distance_signals, peak_details, timestamps, ground_truth)
        
        # SUBPLOT 2: Performance Metrics (Top Right) 
        self._plot_performance_metrics(ax2, metrics, comparison_results)
        
        # SUBPLOT 3: Timeline Comparison (Bottom Left)
        self._plot_timeline_comparison(ax3, comparison_results, ground_truth, detected_timestamps)
        
        # SUBPLOT 4: Error Analysis & Summary (Bottom Right)
        self._plot_error_analysis_summary(ax4, comparison_results, ground_truth, detection_results)
        
        # Add overall title
        video_name = Path(ground_truth['video_path']).stem
        fps = detection_results['processed_fps']
        fig.suptitle(f'Raw MediaPipe Analysis (fps{fps}): {video_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make room for suptitle
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"📊 Raw MediaPipe analysis plot saved: {save_path}")
            plt.close(fig)
    
    def _plot_raw_distance_signals(self, ax, distance_signals, peak_details, timestamps, ground_truth):
        """Plot raw distance signals with peaks and ground truth"""
        colors = {
            'left_hip_heel': '#2E8B57',    # Sea Green
            'right_hip_heel': '#DC143C',   # Crimson
        }
        
        # Plot distance signals
        for signal_name, distances in distance_signals.items():
            if distances is not None:
                color = colors.get(signal_name, 'gray')
                
                # Plot distance signal
                ax.plot(timestamps, distances, 
                       color=color, linestyle='-', 
                       linewidth=2, alpha=0.8,
                       label=signal_name.replace('_', ' ').title())
        
        # Plot detected peaks
        if peak_details['left_peak_indices']:
            left_peak_times = timestamps[peak_details['left_peak_indices']]
            left_peak_values = distance_signals['left_hip_heel'][peak_details['left_peak_indices']]
            ax.scatter(left_peak_times, left_peak_values, color='#2E8B57', s=80, 
                      marker='o', edgecolor='white', linewidth=2, zorder=5, alpha=0.9)
        
        if peak_details['right_peak_indices']:
            right_peak_times = timestamps[peak_details['right_peak_indices']]
            right_peak_values = distance_signals['right_hip_heel'][peak_details['right_peak_indices']]
            ax.scatter(right_peak_times, right_peak_values, color='#DC143C', s=80, 
                      marker='o', edgecolor='white', linewidth=2, zorder=5, alpha=0.9)
        
        # Plot ground truth as vertical lines
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        for timestamp, foot in zip(gt_timestamps, gt_feet):
            line_color = 'blue' if foot == 'left' else 'red'
            ax.axvline(x=timestamp, color=line_color, linestyle=':', 
                      linewidth=2, alpha=0.7, zorder=3)
        
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Raw Hip-Heel Distance (normalized)', fontsize=11)
        ax.set_title('Raw Distance Signals & Peak Detection', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Add processing info
        ax.text(0.02, 0.98, 'RAW MediaPipe\n(No Signal Processing)', 
               transform=ax.transAxes, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
               verticalalignment='top')
    
    def _plot_performance_metrics(self, ax, metrics, comparison_results):
        """Plot performance metrics as bars with summary text"""
        metric_names = ['Detection\nRate', 'Precision', 'F1 Score', 'False Pos.\nRate']
        metric_values = [
            metrics['detection_rate'],
            metrics['precision'], 
            metrics['f1_score'],
            metrics['false_positive_rate']
        ]
        
        colors = ['#2E8B57', '#4169E1', '#9932CC', '#DC143C']
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylim(0, 110)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add summary text box
        summary_text = f"""RAW SUMMARY
Matches: {metrics['total_matches']}/{metrics['total_ground_truth']}
Total Detected: {metrics['total_detected']}
Mean Error: {metrics['mean_abs_error']:.3f}s
Max Error: {metrics['max_abs_error']:.3f}s"""
        
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
               fontfamily='monospace')
    
    def _plot_timeline_comparison(self, ax, comparison_results, ground_truth, detected_timestamps):
        """Plot timeline comparison"""
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        # Plot ground truth
        for i, (timestamp, foot) in enumerate(zip(gt_timestamps, gt_feet)):
            color = 'blue' if foot == 'left' else 'red'
            ax.scatter(timestamp, 2, c=color, s=100, marker='o', alpha=0.8,
                      label='GT Left' if foot == 'left' and i == 0 else 'GT Right' if foot == 'right' and i == 0 else "")
        
        # Plot detected footsteps
        for timestamp in detected_timestamps:
            ax.scatter(timestamp, 1, c='green', s=60, marker='^', alpha=0.8)
        
        # Plot matches with connecting lines
        for match in comparison_results['matches']:
            detected_time = match['detected_timestamp']
            gt_time = match['gt_timestamp']
            ax.plot([detected_time, gt_time], [1, 2], 'gray', alpha=0.5, linewidth=1)
        
        # Highlight false positives and missed detections
        for fp in comparison_results['false_positives']:
            ax.scatter(fp['timestamp'], 1, c='orange', s=60, marker='x', linewidth=3)
        
        for md in comparison_results['missed_detections']:
            ax.scatter(md['timestamp'], 2, facecolor='none', edgecolor='black', 
                      s=100, marker='o', linewidth=2)
        
        ax.set_ylim(0.5, 2.5)
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Detection Level', fontsize=11)
        ax.set_title('Timeline Comparison', fontsize=13, fontweight='bold')
        
        # Custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='GT Left'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='GT Right'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=6, label='Detected'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', markersize=6, label='False +'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                      markeredgecolor='black', markersize=8, label='Missed')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add tolerance info
        tolerance = comparison_results['matching_tolerance']
        ax.text(0.02, 0.98, f'Tolerance: ±{tolerance}s', 
               transform=ax.transAxes, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    def _plot_error_analysis_summary(self, ax, comparison_results, ground_truth, detection_results):
        """Plot error analysis and detailed summary"""
        ax.axis('off')  # Turn off axis for text-only plot
        
        # Compile comprehensive summary
        metrics = comparison_results['metrics']
        
        summary_text = f"""RAW MEDIAPIPE ANALYSIS REPORT

🎯 ACCURACY METRICS:
   Detection Rate:     {metrics['detection_rate']:6.1f}%
   Precision:          {metrics['precision']:6.1f}%
   F1 Score:           {metrics['f1_score']:6.1f}%
   False Positive Rate: {metrics['false_positive_rate']:5.1f}%

⏱️ TIME ACCURACY:
   Mean Absolute Error: {metrics['mean_abs_error']:5.3f}s
   Max Absolute Error:  {metrics['max_abs_error']:5.3f}s
   Mean Bias:          {metrics['mean_time_error']:+5.3f}s
   Standard Deviation: {metrics['std_time_error']:5.3f}s

📊 DETECTION BREAKDOWN:
   Ground Truth Steps:  {metrics['total_ground_truth']:2d}
   Detected Steps:      {metrics['total_detected']:2d}
   Successful Matches:  {metrics['total_matches']:2d}
   False Positives:     {len(comparison_results['false_positives']):2d}
   Missed Detections:   {len(comparison_results['missed_detections']):2d}

🔧 RAW PIPELINE CONFIGURATION:
   Video FPS: {detection_results['video_info']['fps']:.1f}
   Processing FPS: {detection_results['processed_fps']}
   Method: Raw MediaPipe (No Signal Processing)
   Peak Detection: Direct on normalized distances
   
💡 PERFORMANCE ASSESSMENT:"""

        # Add performance assessment
        if metrics['f1_score'] >= 80:
            assessment = "EXCELLENT - Raw approach works well"
        elif metrics['f1_score'] >= 60:
            assessment = "GOOD - Raw approach viable"
        elif metrics['f1_score'] >= 40:
            assessment = "FAIR - Consider signal processing"
        else:
            assessment = "POOR - Signal processing likely needed"
        
        summary_text += f"\n   {assessment}"
        
        # Show individual errors if any
        if comparison_results['missed_detections']:
            summary_text += f"\n\n❌ MISSED DETECTIONS:"
            for i, md in enumerate(comparison_results['missed_detections'][:3]):  # Show max 3
                summary_text += f"\n   {i+1}. {md['timestamp']:.3f}s ({md['foot']})"
            if len(comparison_results['missed_detections']) > 3:
                summary_text += f"\n   ... and {len(comparison_results['missed_detections'])-3} more"
        
        if comparison_results['false_positives']:
            summary_text += f"\n\n⚠️ FALSE POSITIVES:"
            for i, fp in enumerate(comparison_results['false_positives'][:3]):  # Show max 3
                summary_text += f"\n   {i+1}. {fp['timestamp']:.3f}s"
            if len(comparison_results['false_positives']) > 3:
                summary_text += f"\n   ... and {len(comparison_results['false_positives'])-3} more"
        
        # Add text to plot
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        ax.set_title('Raw MediaPipe Analysis Report', fontsize=13, fontweight='bold')

    def analyze_video_with_ground_truth(self, video_path: str, ground_truth_path: str,
                                      save_path: str = None,
                                      verbose: bool = True) -> Dict[str, Any]:
        """
        Complete raw MediaPipe analysis pipeline with combined visualization
        """
        try:
            if verbose:
                print("=" * 60)
                print("RAW MEDIAPIPE COMBINED ANALYSIS")
                print("=" * 60)
            
            # Step 1: Load ground truth
            if verbose:
                print("\n📋 Step 1: Loading ground truth...")
            ground_truth = self.load_ground_truth(ground_truth_path)
            
            if verbose:
                print(f"   Ground truth: {ground_truth['total_steps']} steps")
                print(f"   Left: {ground_truth['left_steps']}, Right: {ground_truth['right_steps']}")
                print(f"   Video duration: {ground_truth['video_info']['duration']:.1f}s")
            
            # Step 2: Run raw MediaPipe detection
            if verbose:
                print(f"\n🔍 Step 2: Running raw MediaPipe detection...")
            
            detector = RawMediaPipeFootstepDetector(
                target_fps=10,  
                confidence_threshold=0.7
            )
            detection_results = detector.process_video(video_path, verbose=verbose)
            
            # Step 3: Compare with ground truth
            if verbose:
                print(f"\n📊 Step 3: Comparing with ground truth...")
            
            comparison_results = self.compare_with_ground_truth(
                detection_results['detected_timestamps'], 
                ground_truth
            )
            
            # Step 4: Print detailed results
            if verbose:
                self.print_detailed_results(comparison_results)
            
            # Step 5: Generate COMBINED visualization
            if verbose:
                print(f"\n📈 Step 5: Creating combined raw MediaPipe visualization...")
            
            if save_path:
                self.create_combined_analysis_plot(
                    detection_results, 
                    ground_truth,
                    comparison_results,
                    save_path
                )
            
            # Compile final results
            analysis_results = {
                'detection_results': detection_results,
                'ground_truth': ground_truth,
                'comparison_results': comparison_results,
                'visualizer_config': self.config
            }
            
            if verbose:
                print(f"\n🎉 Raw MediaPipe analysis complete!")
                if save_path:
                    print(f"   Combined plot saved to: {save_path}")
                
                # Summary
                metrics = comparison_results['metrics']
                print(f"\n📋 Final Summary:")
                print(f"   Detection Rate: {metrics['detection_rate']:.1f}%")
                print(f"   Precision: {metrics['precision']:.1f}%")
                print(f"   F1 Score: {metrics['f1_score']:.1f}%")
                print(f"   Mean Time Error: {metrics['mean_abs_error']:.3f}s")
            
            # Clean up detector
            detector.cleanup()
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error during raw MediaPipe evaluation: {e}")
            raise
    
    def print_detailed_results(self, comparison_results: Dict[str, Any]) -> None:
        """Print detailed comparison results to console"""
        metrics = comparison_results['metrics']
        
        print("\n" + "=" * 60)
        print("RAW MEDIAPIPE EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Detection Rate:     {metrics['detection_rate']:.1f}%")
        print(f"   Precision:          {metrics['precision']:.1f}%") 
        print(f"   F1 Score:           {metrics['f1_score']:.1f}%")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.1f}%")
        
        print(f"\n⏱️ TIME ACCURACY:")
        print(f"   Mean Absolute Error: {metrics['mean_abs_error']:.3f}s")
        print(f"   Max Absolute Error:  {metrics['max_abs_error']:.3f}s")
        print(f"   Mean Error (bias):   {metrics['mean_time_error']:.3f}s")
        print(f"   Std Error:           {metrics['std_time_error']:.3f}s")
        
        print(f"\n🎯 DETECTION SUMMARY:")
        print(f"   Matches:             {metrics['total_matches']}/{metrics['total_ground_truth']}")
        print(f"   Total Detected:      {metrics['total_detected']}")
        print(f"   False Positives:     {len(comparison_results['false_positives'])}")
        print(f"   Missed Detections:   {len(comparison_results['missed_detections'])}")

    def quick_analysis(self, video_path: str, ground_truth_path: str) -> Dict[str, float]:
        """Quick analysis that returns only key metrics"""
        analysis_results = self.analyze_video_with_ground_truth(
            video_path, ground_truth_path, verbose=False
        )
        
        metrics = analysis_results['comparison_results']['metrics']
        
        return {
            'detection_rate': metrics['detection_rate'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'mean_abs_error': metrics['mean_abs_error']
        }


# Batch processing function for raw MediaPipe
def batch_analyze_raw_mediapipe(videos: List[str], test_videos_dir: str = "./test_videos", 
                               output_dir: str = "./output/raw_mediapipe_analysis"):
    """
    Batch analyze videos with raw MediaPipe approach
    Same interface as main visualizer for easy comparison
    """
    
    # Initialize visualizer
    config = RawVisualizerConfig(matching_tolerance=0.3, figure_size=(20, 16))
    visualizer = RawMediaPipeVisualizer(config)
    detector = RawMediaPipeFootstepDetector()
    target_fps = detector.target_fps
    results_summary = {}
    
    print(f"\n🔄 Batch processing {len(videos)} videos with RAW MediaPipe (no display, save only)...")
    print("=" * 70)
    
    for video_name in videos:
        try:
            video_path = f"{test_videos_dir}/{video_name}.mp4"
            ground_truth_path = f"{test_videos_dir}/{video_name}_ground_truth.json"
            combined_plot_path = f"{output_dir}/{video_name}_raw_fps10_complete_analysis.png"
            
            print(f"📹 {video_name} (raw)...", end=" ")
            
            # Full analysis with save (no display)
            analysis_results = visualizer.analyze_video_with_ground_truth(
                video_path=video_path,
                ground_truth_path=ground_truth_path,
                save_path=combined_plot_path,
                verbose=False
            )
            
            # Extract metrics
            metrics = analysis_results['comparison_results']['metrics']
            false_positives_count = len(analysis_results['comparison_results']['false_positives'])

            results_summary[video_name] = {
                'f1_score': metrics['f1_score'],
                'detection_rate': metrics['detection_rate'], 
                'precision': metrics['precision'],
                'mean_abs_error': metrics['mean_abs_error'],
                'processed_fps': target_fps,
                'false_positives': false_positives_count  # Add this
            }
                        
            print(f"✅ F1: {metrics['f1_score']:.1f}% | Saved!")
            
        except FileNotFoundError:
            print(f"❌ Files not found")
            results_summary[video_name] = None
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}...")
            results_summary[video_name] = None
    
    print("=" * 70)
    
    # Print summary table
    print(f"\n📊 RAW PROCESSING RESULTS:")
    print("=" * 90)  # Make wider for additional column
    print(f"{'Video':<8} {'FPS':<5} {'F1 Score':<10} {'Detection':<12} {'Precision':<10} {'False Pos':<10} {'Status':<10}")
    print("-" * 90)
    for video_name, metrics in results_summary.items():
        if metrics:
            fps_display = f"{int(metrics['processed_fps'])}"
            print(f"{video_name:<8} {fps_display:<5} {metrics['f1_score']:>7.1f}% {metrics['detection_rate']:>10.1f}% {metrics['precision']:>9.1f}% {metrics['false_positives']:>8d}   {'SAVED':<10}")
        else:
            print(f"{video_name:<8} {'N/A':<5} {'FAILED':<10} {'FAILED':<12} {'FAILED':<10} {'FAILED':<10}   {'FAILED':<10}")
    print("=" * 90)
    
    print(f"\n✅ All raw MediaPipe plots saved in: {output_dir}/")
    print(f"📊 Files generated:")
    for video_name, metrics in results_summary.items():
        if metrics:
            print(f"   - {video_name}_raw_{metrics['processed_fps']}_complete_analysis.png")
    
    return results_summary


# Main execution function
def main():
    """Run the complete raw MediaPipe batch analysis"""
    print("🎯 RAW MEDIAPIPE BATCH ANALYSIS")
    print("=" * 50)
    print("Pipeline: MediaPipe → Raw Distances → Peak Detection → Alternation")
    print("Skipping: Signal Processor (interpolation + filtering)")
    
    # Test all 5 videos
    videos = ["walk1", "walk2", "walk3", "walk4", "walk5"]
    
    try:
        # Batch analyze all videos
        raw_results = batch_analyze_raw_mediapipe(videos)
        
        print(f"\n🎯 Raw MediaPipe Analysis Complete!")
        print(f"   Processing: No signal filtering")
        print(f"   Peak detection: Direct on raw normalized distances") 
        print(f"   FPS: 10 (same as main pipeline)")
        print(f"   Files saved with '_raw_fps10_' identifier")
        
        # Show best performer
        valid_results = {k: v for k, v in raw_results.items() if v}
        if valid_results:
            best_video = max(valid_results.keys(), key=lambda k: valid_results[k]['f1_score'])
            best_f1 = valid_results[best_video]['f1_score']
            print(f"\n🏆 Best performer: {best_video} (F1: {best_f1:.1f}%)")
        
        return raw_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run batch analysis
    results = main()