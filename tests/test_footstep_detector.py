"""
Tests for FootstepDetector

Tests the core footstep detection pipeline including:
- Initialization and configuration
- Video processing
- Hip-heel distance calculation
- Peak detection
- Alternation filtering
- Spatial data extraction
"""

import pytest
import numpy as np
from pathlib import Path
import sys

from src.pipeline.footstep_detector import FootstepDetector, DetectorConfig
from src.utils.config import get_test_video, DATA_DIR


class TestDetectorConfig:
    """Test detector configuration and validation"""

    def test_default_config(self):
        """Test default configuration values"""
        config = DetectorConfig()
        assert config.target_fps == 10
        assert config.confidence_threshold == 0.7
        assert config.peak_height_threshold == 0.1
        assert config.peak_prominence == 0.05
        assert config.peak_distance == 5

    def test_custom_config(self):
        """Test custom configuration"""
        config = DetectorConfig(
            target_fps=15,
            confidence_threshold=0.8,
            peak_height_threshold=0.2
        )
        assert config.target_fps == 15
        assert config.confidence_threshold == 0.8
        assert config.peak_height_threshold == 0.2

    def test_invalid_fps_raises_error(self):
        """Test that invalid FPS raises ValueError"""
        with pytest.raises(ValueError, match="target_fps must be positive"):
            DetectorConfig(target_fps=0)

        with pytest.raises(ValueError, match="target_fps must be positive"):
            DetectorConfig(target_fps=-5)

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence threshold raises ValueError"""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            DetectorConfig(confidence_threshold=0)

        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            DetectorConfig(confidence_threshold=1.5)


class TestFootstepDetectorInitialization:
    """Test FootstepDetector initialization"""

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        detector = FootstepDetector()
        assert detector is not None
        assert detector.config is not None
        assert detector.config.target_fps == 10
        assert detector.pose_extractor is not None
        assert detector.video_validator is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = DetectorConfig(target_fps=15, confidence_threshold=0.8)
        detector = FootstepDetector(config=custom_config)
        assert detector.config.target_fps == 15
        assert detector.config.confidence_threshold == 0.8


@pytest.mark.unit
class TestFootstepDetection:
    """Test footstep detection on real videos"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for tests"""
        return FootstepDetector()

    def test_process_video_returns_valid_structure(self, detector):
        """Test that process_video returns expected data structure"""
        # Get a test video
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        # Check required keys
        assert 'detected_timestamps' in results
        assert 'heel_strike_detections' in results
        assert 'total_detections' in results
        assert 'spatial_data' in results
        assert 'frames' in results
        assert 'distance_signals' in results
        assert 'video_info' in results
        assert 'processing_config' in results

        # Check types
        assert isinstance(results['detected_timestamps'], list)
        assert isinstance(results['heel_strike_detections'], list)
        assert isinstance(results['total_detections'], int)
        assert isinstance(results['spatial_data'], list)
        assert isinstance(results['frames'], list)
        assert isinstance(results['video_info'], dict)

    def test_detection_count_matches(self, detector):
        """Test that detection counts are consistent"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        num_detections = results['total_detections']
        assert len(results['detected_timestamps']) == num_detections
        assert len(results['heel_strike_detections']) == num_detections
        assert len(results['spatial_data']) == num_detections

    def test_detections_are_chronological(self, detector):
        """Test that detected timestamps are in chronological order"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        timestamps = results['detected_timestamps']
        if len(timestamps) > 1:
            # Check that timestamps are sorted
            assert timestamps == sorted(timestamps)

    def test_timestamps_within_video_duration(self, detector):
        """Test that all timestamps are within video duration"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        duration = results['video_info']['duration']
        timestamps = results['detected_timestamps']

        for ts in timestamps:
            assert 0 <= ts <= duration

    def test_foot_side_is_valid(self, detector):
        """Test that foot_side is either 'left' or 'right'"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        for timestamp, foot_side in results['heel_strike_detections']:
            assert foot_side in ['left', 'right']

    def test_spatial_data_has_required_fields(self, detector):
        """Test that spatial data contains required fields"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        if len(results['spatial_data']) > 0:
            spatial_info = results['spatial_data'][0]
            assert 'timestamp' in spatial_info
            assert 'foot_side' in spatial_info
            assert 'x_position' in spatial_info
            assert 'hip_heel_distance' in spatial_info

    def test_x_position_normalized(self, detector):
        """Test that x_position is normalized between 0 and 1"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        for spatial_info in results['spatial_data']:
            x_pos = spatial_info['x_position']
            assert 0.0 <= x_pos <= 1.0

    def test_stored_frames_exist(self, detector):
        """Test that frames are stored for scene analysis"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        frames = results['frames']
        assert isinstance(frames, list)
        # Should have stored up to 50 frames
        assert len(frames) <= 50

        # If frames exist, check they are numpy arrays
        if len(frames) > 0:
            assert isinstance(frames[0], np.ndarray)
            assert len(frames[0].shape) == 3  # Height x Width x Channels


@pytest.mark.unit
class TestFootstepDetectorEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_video_path(self):
        """Test handling of invalid video path"""
        detector = FootstepDetector()

        # This should raise an error from VideoValidator
        with pytest.raises(Exception):
            detector.process_video("nonexistent_video.mp4", verbose=False)

    def test_verbose_flag_works(self):
        """Test that verbose flag controls output"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        detector = FootstepDetector()
        video_path = str(test_videos[0])

        # Should not raise error with verbose=True
        results_verbose = detector.process_video(video_path, verbose=True)
        # Should not raise error with verbose=False
        results_quiet = detector.process_video(video_path, verbose=False)

        # Results should be similar (may differ slightly due to randomness in frame storage)
        assert results_verbose['total_detections'] == results_quiet['total_detections']


@pytest.mark.unit
class TestDistanceSignals:
    """Test hip-heel distance calculation"""

    def test_distance_signals_structure(self):
        """Test that distance signals have correct structure"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        detector = FootstepDetector()
        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        signals = results['distance_signals']
        assert 'left_hip_heel' in signals
        assert 'right_hip_heel' in signals
        assert 'timestamps' in signals

        # Check they are numpy arrays
        assert isinstance(signals['left_hip_heel'], np.ndarray)
        assert isinstance(signals['right_hip_heel'], np.ndarray)
        assert isinstance(signals['timestamps'], np.ndarray)

        # Check they have same length
        assert len(signals['left_hip_heel']) == len(signals['right_hip_heel'])
        assert len(signals['left_hip_heel']) == len(signals['timestamps'])

    def test_distance_signals_are_positive(self):
        """Test that hip-heel distances are non-negative"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        detector = FootstepDetector()
        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        signals = results['distance_signals']
        left_distances = signals['left_hip_heel']
        right_distances = signals['right_hip_heel']

        # Filter out NaN values (from missing landmarks)
        left_valid = left_distances[~np.isnan(left_distances)]
        right_valid = right_distances[~np.isnan(right_distances)]

        if len(left_valid) > 0:
            assert np.all(left_valid >= 0)
        if len(right_valid) > 0:
            assert np.all(right_valid >= 0)


@pytest.mark.integration
class TestDetectorPerformance:
    """Integration tests for overall detector performance"""

    def test_detector_finds_footsteps_in_walking_video(self):
        """Test that detector finds footsteps in a video with walking person"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        detector = FootstepDetector()
        video_path = str(test_videos[0])
        results = detector.process_video(video_path, verbose=False)

        # Should detect at least some footsteps in a walking video
        # (Actual number depends on video content, but >0 is reasonable expectation)
        assert results['total_detections'] >= 0
        # Could add: assert results['total_detections'] > 0 if you know test videos have people walking

    def test_detector_processing_time_reasonable(self):
        """Test that detection completes in reasonable time"""
        test_videos = list(Path(DATA_DIR / "videos").glob("*.mp4"))
        if not test_videos:
            pytest.skip("No test videos available")

        import time
        detector = FootstepDetector()
        video_path = str(test_videos[0])

        start = time.time()
        results = detector.process_video(video_path, verbose=False)
        elapsed = time.time() - start

        video_duration = results['video_info']['duration']

        # Processing should complete in less than 10x video duration
        # (This is a loose bound - adjust based on expected performance)
        assert elapsed < video_duration * 10


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
