"""
Tests for SpatialAudioProcessor

Tests spatial audio processing including:
- Initialization and configuration
- Panning calculations (constant power)
- Attenuation calculations (inverse distance law)
- Audio chopping and segment extraction
- Full spatial audio pipeline
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from src.pipeline.spatial_audio_processor import SpatialAudioProcessor


class TestSpatialAudioProcessorInitialization:
    """Test SpatialAudioProcessor initialization"""

    def test_init_with_default_params(self):
        """Test initialization with default parameters"""
        processor = SpatialAudioProcessor()
        assert processor.sample_rate == 44100
        assert processor.fade_duration == 0.01
        assert processor.min_attenuation_db == -20.0
        assert processor.max_attenuation_db == 0.0
        assert processor.max_pan_percentage == 0.2
        assert processor.distance_curve_power == 2.0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters"""
        processor = SpatialAudioProcessor(
            sample_rate=48000,
            fade_duration=0.02,
            min_attenuation_db=-30.0,
            max_attenuation_db=-3.0,
            max_pan_percentage=0.4,
            distance_curve_power=1.5
        )
        assert processor.sample_rate == 48000
        assert processor.fade_duration == 0.02
        assert processor.min_attenuation_db == -30.0
        assert processor.max_attenuation_db == -3.0
        assert processor.max_pan_percentage == 0.4
        assert processor.distance_curve_power == 1.5


@pytest.mark.unit
class TestPanningCalculations:
    """Test constant power panning calculations"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    def test_center_panning_equal_gains(self, processor):
        """Test that center position (0.5) gives equal L/R gains"""
        # Center position should give equal left and right gains
        left_gain, right_gain = processor._calculate_pan_gains(0.5)

        # With constant power panning, center should be approximately equal
        assert abs(left_gain - right_gain) < 0.01

    def test_full_left_panning(self, processor):
        """Test that left position (0.0) gives more left gain"""
        left_gain, right_gain = processor._calculate_pan_gains(0.0)

        # Left position should favor left channel
        assert left_gain > right_gain

    def test_full_right_panning(self, processor):
        """Test that right position (1.0) gives more right gain"""
        left_gain, right_gain = processor._calculate_pan_gains(1.0)

        # Right position should favor right channel
        assert right_gain > left_gain

    def test_constant_power_sum(self, processor):
        """Test that constant power panning maintains power"""
        # Constant power panning should satisfy: left^2 + right^2 ≈ constant
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        powers = []

        for pos in positions:
            left_gain, right_gain = processor._calculate_pan_gains(pos)
            power = left_gain**2 + right_gain**2
            powers.append(power)

        # All powers should be approximately equal
        assert np.std(powers) < 0.1  # Small variation allowed

    def test_pan_percentage_limits_range(self):
        """Test that max_pan_percentage limits panning range"""
        # Subtle panning (20%)
        processor_subtle = SpatialAudioProcessor(max_pan_percentage=0.2)
        left_s, right_s = processor_subtle._calculate_pan_gains(0.0)

        # Aggressive panning (100%)
        processor_aggressive = SpatialAudioProcessor(max_pan_percentage=1.0)
        left_a, right_a = processor_aggressive._calculate_pan_gains(0.0)

        # Aggressive should have larger difference
        diff_subtle = abs(left_s - right_s)
        diff_aggressive = abs(left_a - right_a)
        assert diff_aggressive > diff_subtle


@pytest.mark.unit
class TestAttenuationCalculations:
    """Test inverse distance attenuation calculations"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    def test_closest_distance_no_attenuation(self, processor):
        """Test that closest distance (reference) has minimal attenuation"""
        reference_distance = 100.0
        attenuation_db = processor._calculate_attenuation(
            distance=reference_distance,
            reference_distance=reference_distance
        )

        # Closest distance should have max attenuation (0dB or close)
        assert attenuation_db >= processor.max_attenuation_db - 1.0

    def test_farther_distance_more_attenuation(self, processor):
        """Test that farther distances have more attenuation"""
        reference_distance = 100.0

        close_attenuation = processor._calculate_attenuation(150.0, reference_distance)
        far_attenuation = processor._calculate_attenuation(300.0, reference_distance)

        # Farther should be more attenuated (more negative dB)
        assert far_attenuation < close_attenuation

    def test_inverse_square_law(self):
        """Test that distance_curve_power=2.0 gives inverse square law"""
        processor = SpatialAudioProcessor(
            distance_curve_power=2.0,
            min_attenuation_db=-20.0
        )
        reference_distance = 100.0

        # Double the distance
        attenuation_1x = processor._calculate_attenuation(100.0, reference_distance)
        attenuation_2x = processor._calculate_attenuation(200.0, reference_distance)

        # With inverse square law, doubling distance should reduce by ~6dB
        # (Not exact due to clamping and normalization, but should be in ballpark)
        db_reduction = attenuation_1x - attenuation_2x
        assert 4.0 < db_reduction < 8.0  # Approximately 6dB

    def test_attenuation_clamped_to_range(self, processor):
        """Test that attenuation is clamped to min/max range"""
        reference_distance = 100.0

        # Very far distance
        very_far_attenuation = processor._calculate_attenuation(10000.0, reference_distance)

        # Should be clamped to minimum
        assert very_far_attenuation >= processor.min_attenuation_db
        assert very_far_attenuation <= processor.max_attenuation_db


@pytest.mark.unit
class TestAudioSegmentation:
    """Test audio chopping and segmentation"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    def test_chop_audio_returns_segments(self, processor):
        """Test that audio chopping returns list of segments"""
        # Create synthetic audio with distinct footstep-like sounds
        sr = 44100
        audio = self._create_synthetic_footsteps(sr, num_footsteps=3)

        segments = processor._chop_audio_at_quiet_zones(audio, sr)

        assert isinstance(segments, list)
        assert len(segments) > 0

    def test_chopped_segments_are_arrays(self, processor):
        """Test that chopped segments are numpy arrays"""
        sr = 44100
        audio = self._create_synthetic_footsteps(sr, num_footsteps=3)
        segments = processor._chop_audio_at_quiet_zones(audio, sr)

        for segment in segments:
            assert isinstance(segment, np.ndarray)
            assert len(segment) > 0

    def _create_synthetic_footsteps(self, sr, num_footsteps=3, duration=6.0):
        """Helper: Create synthetic audio with footstep-like sounds"""
        total_samples = int(sr * duration)
        audio = np.zeros(total_samples)

        # Add footstep-like impulses with decay
        for i in range(num_footsteps):
            position = int((i + 0.5) * total_samples / num_footsteps)
            # Short impulse with exponential decay
            impulse_length = int(0.1 * sr)
            if position + impulse_length < len(audio):
                t = np.arange(impulse_length) / sr
                impulse = np.exp(-10 * t) * np.sin(2 * np.pi * 200 * t)
                audio[position:position + impulse_length] = impulse

        return audio


@pytest.mark.unit
class TestSpatialMixing:
    """Test spatial audio mixing"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    @pytest.fixture
    def mock_detection_results(self):
        """Create mock detection results for testing"""
        return {
            'video_info': {
                'duration': 5.0,
                'width': 1920,
                'fps': 30.0
            },
            'spatial_data': [
                {
                    'timestamp': 0.5,
                    'foot_side': 'left',
                    'x_position': 0.3,
                    'hip_heel_distance': 150.0
                },
                {
                    'timestamp': 1.2,
                    'foot_side': 'right',
                    'x_position': 0.7,
                    'hip_heel_distance': 120.0
                },
                {
                    'timestamp': 2.0,
                    'foot_side': 'left',
                    'x_position': 0.4,
                    'hip_heel_distance': 180.0
                }
            ]
        }

    def test_create_spatial_mix_returns_stereo(self, processor, mock_detection_results):
        """Test that spatial mixing creates stereo output"""
        sr = 44100
        segments = [self._create_segment(sr, 0.2) for _ in range(3)]

        spatial_data = mock_detection_results['spatial_data']
        reference_distance = 100.0
        video_duration = mock_detection_results['video_info']['duration']
        video_width = mock_detection_results['video_info']['width']

        stereo_audio = processor._create_spatial_mix(
            segments, spatial_data, reference_distance,
            video_duration, video_width, sr
        )

        # Should return stereo array (2D: 2 channels x samples)
        assert stereo_audio.shape[0] == 2
        assert stereo_audio.shape[1] > 0

    def test_spatial_mix_matches_duration(self, processor, mock_detection_results):
        """Test that mixed audio matches target duration"""
        sr = 44100
        segments = [self._create_segment(sr, 0.2) for _ in range(3)]

        spatial_data = mock_detection_results['spatial_data']
        reference_distance = 100.0
        video_duration = mock_detection_results['video_info']['duration']
        video_width = mock_detection_results['video_info']['width']

        stereo_audio = processor._create_spatial_mix(
            segments, spatial_data, reference_distance,
            video_duration, video_width, sr
        )

        expected_samples = int(video_duration * sr)
        actual_samples = stereo_audio.shape[1]

        # Should match duration (within small tolerance)
        assert abs(actual_samples - expected_samples) < sr * 0.1  # Within 0.1 seconds

    def _create_segment(self, sr, duration=0.2):
        """Helper: Create a short audio segment"""
        samples = int(sr * duration)
        t = np.arange(samples) / sr
        # Decaying sine wave
        segment = np.exp(-10 * t) * np.sin(2 * np.pi * 200 * t)
        return segment


@pytest.mark.unit
class TestFinalProcessing:
    """Test final audio processing (fades, normalization)"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    def test_apply_final_processing_returns_correct_shape(self, processor):
        """Test that final processing maintains stereo shape"""
        sr = 44100
        duration = 5.0
        samples = int(sr * duration)

        # Create stereo audio
        audio = np.random.randn(2, samples) * 0.1

        processed = processor._apply_final_processing(audio, duration, sr)

        assert processed.shape == audio.shape

    def test_normalization_peak_level(self, processor):
        """Test that audio is normalized to correct peak level"""
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)

        # Create audio with known peak
        audio = np.random.randn(2, samples) * 0.5

        processed = processor._apply_final_processing(audio, duration, sr)

        # Peak should be close to -6dB (0.5 amplitude)
        peak = np.max(np.abs(processed))
        expected_peak = 0.5  # -6dB

        assert abs(peak - expected_peak) < 0.1


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for complete spatial audio pipeline"""

    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return SpatialAudioProcessor()

    @pytest.fixture
    def mock_audio_file(self, tmp_path):
        """Create a temporary mock audio file"""
        sr = 44100
        duration = 6.0
        samples = int(sr * duration)

        # Create audio with footstep-like sounds
        audio = np.zeros(samples)
        for i in range(3):
            position = int((i + 0.5) * samples / 3)
            impulse_length = int(0.1 * sr)
            if position + impulse_length < len(audio):
                t = np.arange(impulse_length) / sr
                impulse = np.exp(-10 * t) * np.sin(2 * np.pi * 200 * t)
                audio[position:position + impulse_length] = impulse

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), audio, sr)
        return str(audio_path)

    @pytest.fixture
    def mock_detection_results(self):
        """Create mock detection results"""
        return {
            'video_info': {
                'duration': 5.0,
                'width': 1920,
                'fps': 30.0
            },
            'spatial_data': [
                {
                    'timestamp': 0.5,
                    'foot_side': 'left',
                    'x_position': 0.3,
                    'hip_heel_distance': 150.0
                },
                {
                    'timestamp': 1.5,
                    'foot_side': 'right',
                    'x_position': 0.7,
                    'hip_heel_distance': 120.0
                },
                {
                    'timestamp': 2.5,
                    'foot_side': 'left',
                    'x_position': 0.4,
                    'hip_heel_distance': 180.0
                }
            ]
        }

    def test_full_pipeline_with_file_input(self, processor, mock_audio_file,
                                           mock_detection_results, tmp_path):
        """Test complete pipeline with file input"""
        output_path = tmp_path / "output.wav"

        result = processor.process_video_audio(
            audio_input=mock_audio_file,
            detection_results=mock_detection_results,
            output_path=str(output_path),
            visualize=False
        )

        # Check output file was created
        assert output_path.exists()

        # Check result structure
        assert 'output_path' in result
        assert 'processing_stats' in result

    def test_full_pipeline_with_array_input(self, processor, mock_detection_results, tmp_path):
        """Test complete pipeline with numpy array input"""
        sr = 44100
        duration = 6.0
        audio = np.random.randn(int(sr * duration)) * 0.1

        output_path = tmp_path / "output.wav"

        result = processor.process_video_audio(
            audio_input=(audio, sr),
            detection_results=mock_detection_results,
            output_path=str(output_path),
            visualize=False
        )

        # Check output file was created
        assert output_path.exists()

        # Load and verify output
        output_audio, output_sr = sf.read(str(output_path))
        assert output_sr == sr
        assert len(output_audio.shape) == 2  # Stereo
        assert output_audio.shape[0] > 0

    def test_output_audio_is_stereo(self, processor, mock_detection_results, tmp_path):
        """Test that output audio is stereo"""
        sr = 44100
        audio = np.random.randn(int(sr * 6.0)) * 0.1
        output_path = tmp_path / "output.wav"

        processor.process_video_audio(
            audio_input=(audio, sr),
            detection_results=mock_detection_results,
            output_path=str(output_path),
            visualize=False
        )

        output_audio, _ = sf.read(str(output_path))

        # Should be 2D array (samples x 2 channels)
        assert len(output_audio.shape) == 2
        assert output_audio.shape[1] == 2

    def test_output_duration_matches_video(self, processor, mock_detection_results, tmp_path):
        """Test that output duration matches video duration"""
        sr = 44100
        audio = np.random.randn(int(sr * 6.0)) * 0.1
        output_path = tmp_path / "output.wav"

        processor.process_video_audio(
            audio_input=(audio, sr),
            detection_results=mock_detection_results,
            output_path=str(output_path),
            visualize=False
        )

        output_audio, output_sr = sf.read(str(output_path))
        output_duration = len(output_audio) / output_sr
        expected_duration = mock_detection_results['video_info']['duration']

        # Should match within 0.1 seconds
        assert abs(output_duration - expected_duration) < 0.1


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_spatial_data(self):
        """Test handling of empty spatial data"""
        processor = SpatialAudioProcessor()

        detection_results = {
            'video_info': {'duration': 5.0, 'width': 1920, 'fps': 30.0},
            'spatial_data': []  # No detections
        }

        sr = 44100
        audio = np.random.randn(int(sr * 6.0)) * 0.1

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name

        # Should handle empty detections gracefully
        try:
            result = processor.process_video_audio(
                audio_input=(audio, sr),
                detection_results=detection_results,
                output_path=output_path,
                visualize=False
            )
            # Should still create output file (silent or minimal audio)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_single_detection(self):
        """Test handling of single detection"""
        processor = SpatialAudioProcessor()

        detection_results = {
            'video_info': {'duration': 5.0, 'width': 1920, 'fps': 30.0},
            'spatial_data': [
                {
                    'timestamp': 2.5,
                    'foot_side': 'left',
                    'x_position': 0.5,
                    'hip_heel_distance': 150.0
                }
            ]
        }

        sr = 44100
        audio = np.random.randn(int(sr * 6.0)) * 0.1

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name

        try:
            result = processor.process_video_audio(
                audio_input=(audio, sr),
                detection_results=detection_results,
                output_path=output_path,
                visualize=False
            )
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
