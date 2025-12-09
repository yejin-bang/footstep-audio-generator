import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 🔧 Fixed import path
from archive.full_footstep_detector import FootstepDetector, FootstepDetectorConfig


@dataclass
class VisualizerConfig:
    """Configuration for visualization and evaluation"""
    # Matching parameters
    matching_tolerance: float = 0.3  # ±0.3 seconds for ground truth matching
    
    # Visualization parameters - Updated for combined plot
    figure_size: Tuple[int, int] = (20, 16)  # Larger for combined view
    dpi: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.matching_tolerance <= 0:
            raise ValueError("matching_tolerance must be positive")


class FootstepVisualizer:
    """
    Visualization and evaluation module for footstep detection
    
    NEW: All plots combined into a single comprehensive visualization
    """
    
    def __init__(self, config: Optional[VisualizerConfig] = None):
        """Initialize footstep visualizer"""
        self.config = config if config is not None else VisualizerConfig()
        
        # Storage for analysis results
        self.last_ground_truth = None
        self.last_comparison_results = None
        self.last_detection_results = None
    
    def load_ground_truth(self, json_path: str) -> Dict[str, Any]:
        """Load ground truth annotations from JSON file"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            ground_truth_data = json.load(f)
        
        # Extract timestamps and foot information
        gt_timestamps = []
        gt_feet = []
        gt_frames = []
        
        for annotation in ground_truth_data['annotations']:
            gt_timestamps.append(annotation['timestamp'])
            gt_feet.append(annotation['foot'])
            gt_frames.append(annotation['frame'])
        
        processed_gt = {
            'timestamps': np.array(gt_timestamps),
            'feet': gt_feet,
            'frames': np.array(gt_frames),
            'total_steps': ground_truth_data['summary']['total_steps'],
            'left_steps': ground_truth_data['summary']['left_steps'],
            'right_steps': ground_truth_data['summary']['right_steps'],
            'video_info': ground_truth_data['video_info'],
            'video_path': ground_truth_data['video_path']
        }
        
        return processed_gt
    
    def compare_with_ground_truth(self, detected_timestamps: List[float], 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare detected footsteps with ground truth annotations"""
        gt_timestamps = ground_truth['timestamps']
        tolerance = self.config.matching_tolerance
        
        # Find matches between detected and ground truth
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
        
        comparison_results = {
            'matches': matches,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'metrics': metrics,
            'matching_tolerance': tolerance,
            'total_detected': len(detected_timestamps),
            'total_ground_truth': len(gt_timestamps)
        }
        
        return comparison_results
    
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
        
        metrics = {
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
        
        return metrics

    def create_combined_analysis_plot(self, detection_results: Dict[str, Any], 
                                    ground_truth: Dict[str, Any],
                                    comparison_results: Dict[str, Any],
                                    save_path: str = None) -> None:
        """
        NEW: Create comprehensive combined plot with all analysis in one figure
        
        Layout:
        +---------------------------+---------------------------+
        |                           |                           |
        |    Distance Signals       |    Performance Metrics    |
        |    (with GT & peaks)      |    (bars + text summary)  |
        |                           |                           |
        +---------------------------+---------------------------+
        |                           |                           |
        |    Timeline Comparison    |    Error Analysis         |
        |    (GT vs Detected)       |    (detailed breakdown)   |
        |                           |                           |
        +---------------------------+---------------------------+
        """
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size, 
                                                     dpi=self.config.dpi)
        
        # Get data
        gait_results = detection_results['gait_results']
        distance_signals = gait_results['distance_signals']
        detailed_results = gait_results['detailed_results']
        timestamps = detection_results['processed_data']['timestamps']
        detected_timestamps = detection_results['detected_timestamps']
        metrics = comparison_results['metrics']
        
        # SUBPLOT 1: Distance Signals (Top Left)
        self._plot_distance_signals(ax1, distance_signals, detailed_results, timestamps, ground_truth)
        
        # SUBPLOT 2: Performance Metrics (Top Right) 
        self._plot_performance_metrics(ax2, metrics, comparison_results)
        
        # SUBPLOT 3: Timeline Comparison (Bottom Left)
        self._plot_timeline_comparison(ax3, comparison_results, ground_truth, detected_timestamps)
        
        # SUBPLOT 4: Error Analysis & Summary (Bottom Right)
        self._plot_error_analysis_summary(ax4, comparison_results, ground_truth, detection_results)
        
        # Add overall title
        video_name = Path(ground_truth['video_path']).stem
        fig.suptitle(f'Complete Footstep Detection Analysis: {video_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make room for suptitle
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"📊 Combined analysis plot saved: {save_path}")
            # Close the figure to free memory and avoid display
            plt.close(fig)
    
    def _plot_distance_signals(self, ax, distance_signals, detailed_results, timestamps, ground_truth):
        """Plot distance signals with peaks and ground truth"""
        colors = {
            'left_hip_heel': '#2E8B57',    # Sea Green
            'right_hip_heel': '#DC143C',   # Crimson
        }
        
        # Plot distance signals
        signal_legends = []
        for signal_name, distances in distance_signals.items():
            if distances is not None:
                color = colors.get(signal_name, 'gray')
                
                # Plot distance signal
                line = ax.plot(timestamps, distances, 
                              color=color, linestyle='-', 
                              linewidth=2, alpha=0.8,
                              label=signal_name.replace('_', ' ').title())[0]
                signal_legends.append(line)
                
                # Plot detected peaks
                if signal_name in detailed_results:
                    peak_indices = detailed_results[signal_name]['peak_frame_indices']
                    if peak_indices:
                        peak_timestamps = timestamps[peak_indices]
                        peak_values = distances[peak_indices]
                        
                        ax.scatter(peak_timestamps, peak_values, 
                                  color=color, s=80, marker='o', 
                                  edgecolor='white', linewidth=2,
                                  zorder=5, alpha=0.9)
        
        # Plot ground truth as vertical lines
        gt_timestamps = ground_truth['timestamps']
        gt_feet = ground_truth['feet']
        
        for timestamp, foot in zip(gt_timestamps, gt_feet):
            line_color = 'blue' if foot == 'left' else 'red'
            ax.axvline(x=timestamp, color=line_color, linestyle=':', 
                      linewidth=2, alpha=0.7, zorder=3)
        
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Hip-Heel Distance (normalized)', fontsize=11)
        ax.set_title('Distance Signals & Peak Detection', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
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
        summary_text = f"""SUMMARY
Matches: {metrics['total_matches']}/{metrics['total_ground_truth']}
Total Detected: {metrics['total_detected']}
Mean Error: {metrics['mean_abs_error']:.3f}s
Max Error: {metrics['max_abs_error']:.3f}s"""
        
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
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
        
        # Pipeline info
        pipeline_info = detection_results.get('pipeline_metadata', {})
        gait_config = pipeline_info.get('gait_detector_config', {})
        
        summary_text = f"""DETECTION ANALYSIS REPORT

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

🔧 PIPELINE CONFIGURATION:
   Video FPS: {detection_results['video_info']['fps']:.1f}
   Processing FPS: {pipeline_info.get('processed_fps', 'N/A')}
   Gait Detector: PureAlternationGaitDetector
   
💡 PERFORMANCE ASSESSMENT:"""

        # Add performance assessment
        if metrics['f1_score'] >= 80:
            assessment = "EXCELLENT - Ready for production"
        elif metrics['f1_score'] >= 60:
            assessment = "GOOD - Minor tuning recommended"
        elif metrics['f1_score'] >= 40:
            assessment = "FAIR - Needs improvement"
        else:
            assessment = "POOR - Major issues detected"
        
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
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        ax.set_title('Detailed Analysis Report', fontsize=13, fontweight='bold')
    
    def print_detailed_results(self, comparison_results: Dict[str, Any]) -> None:
        """Print detailed comparison results to console"""
        metrics = comparison_results['metrics']
        
        print("\n" + "=" * 60)
        print("FOOTSTEP DETECTION EVALUATION RESULTS")
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
    
    def analyze_video_with_ground_truth(self, video_path: str, ground_truth_path: str,
                                      save_path: str = None,
                                      verbose: bool = True) -> Dict[str, Any]:
        """
        Complete analysis pipeline with NEW combined visualization
        
        Args:
            video_path: Path to video file
            ground_truth_path: Path to ground truth JSON
            save_path: Path to save combined plot (replaces separate plot paths)
            verbose: Print detailed information
        """
        try:
            if verbose:
                print("=" * 60)
                print("COMBINED FOOTSTEP DETECTION ANALYSIS")
                print("=" * 60)
            
            # Step 1: Load ground truth
            if verbose:
                print("\n📋 Step 1: Loading ground truth...")
            ground_truth = self.load_ground_truth(ground_truth_path)
            self.last_ground_truth = ground_truth
            
            if verbose:
                print(f"   Ground truth: {ground_truth['total_steps']} steps")
                print(f"   Left: {ground_truth['left_steps']}, Right: {ground_truth['right_steps']}")
                print(f"   Video duration: {ground_truth['video_info']['duration']:.1f}s")
            
            # Step 2: Run detection pipeline
            if verbose:
                print(f"\n🔍 Step 2: Running detection pipeline...")
            
            detector = FootstepDetector()
            detection_results = detector.process_video(video_path, verbose=verbose)
            self.last_detection_results = detection_results
            
            # Step 3: Compare with ground truth
            if verbose:
                print(f"\n📊 Step 3: Comparing with ground truth...")
            
            comparison_results = self.compare_with_ground_truth(
                detection_results['detected_timestamps'], 
                ground_truth
            )
            self.last_comparison_results = comparison_results
            
            # Step 4: Print detailed results
            if verbose:
                self.print_detailed_results(comparison_results)
            
            # Step 5: Generate COMBINED visualization
            if verbose:
                print(f"\n📈 Step 5: Creating combined analysis visualization...")
            
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
                print(f"\n🎉 Combined analysis complete!")
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
            print(f"❌ Error during evaluation: {e}")
            raise
    
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


# Example usage and testing
if __name__ == "__main__":
    # Configuration for combined visualization
    config = VisualizerConfig(
        matching_tolerance=0.3,
        figure_size=(20, 16)  # Large size for combined plot
    )

    visualizer = FootstepVisualizer(config)

    videos = ["walk1", "walk2", "walk3", "walk4", "walk5"]
    results_summary = {}

    print(f"\n🔄 Batch processing all {len(videos)} videos (no display, save only)...")
    print("=" * 60)

    for video_name in videos:
        try:
            video_path = f"./test_videos/{video_name}.mp4"
            ground_truth_path = f"./test_videos/{video_name}_ground_truth.json"
            
            # Get target FPS from the actual configuration
            config = FootstepDetectorConfig()
            target_fps = config.pose_extractor_fps
            combined_plot_path = f"./output/combined_analysis/{video_name}_fps{target_fps}_complete_analysis.png"
            
            print(f"📹 {video_name} (fps{target_fps})...", end=" ")
            
            # SINGLE CALL - analysis and save together
            analysis_results = visualizer.analyze_video_with_ground_truth(
                video_path=video_path,
                ground_truth_path=ground_truth_path,
                save_path=combined_plot_path,
                verbose=False
            )
            
            # Extract and store metrics
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
            
            print(f" ✅ F1: {metrics['f1_score']:.1f}% | Saved!")
            
        except FileNotFoundError:
            print(f"📹 {video_name}... ❌ Files not found")
            results_summary[video_name] = None
        except Exception as e:
            print(f"📹 {video_name}... ❌ Error: {str(e)[:30]}...")
            results_summary[video_name] = None

    print("=" * 60)
    # Print summary table
    print(f"\n📊 FULL PROCESSING RESULTS:")
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

    print(f"\n✅ All plots saved in: ./output/combined_analysis/")
    print(f"📊 Files generated:")
    for video_name, metrics in results_summary.items():
        if metrics:
            fps_int = int(metrics['processed_fps'])
            print(f"   - {video_name}_fps{fps_int}_complete_analysis.png")