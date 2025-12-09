"""
Unit Tests for Audio Generation Backends

Tests pluggable audio backend system.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_backends import get_backend, list_backends, MockBackend
from src.audio_backends.base import AudioBackend


@pytest.mark.unit
class TestBackendFactory:
    """Test backend factory functions."""

    def test_list_backends(self):
        """Test that list_backends returns available backends."""
        backends = list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
        assert "mock" in backends
        assert "runpod" in backends

    def test_get_backend_mock(self):
        """Test getting mock backend."""
        backend = get_backend("mock")
        assert isinstance(backend, MockBackend)
        assert isinstance(backend, AudioBackend)

    def test_get_backend_invalid_raises(self):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_backend("nonexistent_backend")

        error_msg = str(exc_info.value)
        assert "nonexistent_backend" in error_msg
        assert "Available backends" in error_msg


@pytest.mark.unit
class TestMockBackend:
    """Test MockBackend functionality."""

    def test_mock_backend_initialization(self):
        """Test MockBackend initializes correctly."""
        backend = MockBackend(mode="footsteps", sample_rate=44100)
        assert backend.mode == "footsteps"
        assert backend.sample_rate == 44100

    def test_mock_backend_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MockBackend(mode="invalid_mode")

        assert "invalid_mode" in str(exc_info.value)

    def test_mock_backend_generate_returns_correct_format(self):
        """Test that generate() returns correct data format."""
        backend = MockBackend(mode="footsteps")

        audio, sample_rate, metadata = backend.generate(
            prompt="test prompt",
            audio_length=2.0
        )

        # Check audio array
        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 2  # Stereo (2 channels)
        assert audio.dtype == np.float32

        # Check sample rate
        assert isinstance(sample_rate, int)
        assert sample_rate == backend.sample_rate

        # Check audio length (approximately 2 seconds)
        expected_samples = int(2.0 * sample_rate)
        assert abs(audio.shape[1] - expected_samples) < 100  # Allow small tolerance

        # Check metadata
        assert isinstance(metadata, dict)
        assert metadata["backend"] == "mock"
        assert metadata["mode"] == "footsteps"
        assert metadata["prompt"] == "test prompt"
        assert "sample_rate" in metadata
        assert "channels" in metadata

    def test_mock_backend_different_modes(self):
        """Test different generation modes."""
        modes = ["sine", "noise", "silence", "footsteps"]

        for mode in modes:
            backend = MockBackend(mode=mode)
            audio, sr, metadata = backend.generate("test", audio_length=1.0)

            assert audio.shape[0] == 2  # Stereo
            assert metadata["mode"] == mode

            # Check that different modes produce different audio
            if mode == "silence":
                assert np.allclose(audio, 0.0)
            else:
                assert not np.allclose(audio, 0.0)

    def test_mock_backend_get_info(self):
        """Test get_info() method."""
        backend = MockBackend(mode="footsteps")
        info = backend.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "MockBackend"
        assert info["type"] == "synthetic"
        assert info["mode"] == "footsteps"
        assert info["requires_api_key"] == False

    def test_mock_backend_repr(self):
        """Test string representation."""
        backend = MockBackend(mode="footsteps", sample_rate=44100)
        repr_str = repr(backend)

        assert "MockBackend" in repr_str
        assert "footsteps" in repr_str
        assert "44100" in repr_str


@pytest.mark.unit
class TestAudioBackendInterface:
    """Test that backends implement the required interface."""

    def test_mock_backend_has_generate_method(self):
        """Test that MockBackend has generate() method."""
        backend = MockBackend()
        assert hasattr(backend, 'generate')
        assert callable(backend.generate)

    def test_mock_backend_has_get_info_method(self):
        """Test that MockBackend has get_info() method."""
        backend = MockBackend()
        assert hasattr(backend, 'get_info')
        assert callable(backend.get_info)

    def test_mock_backend_generate_signature(self):
        """Test that generate() accepts required parameters."""
        backend = MockBackend()

        # Should work with just prompt
        audio, sr, metadata = backend.generate("test")
        assert audio is not None

        # Should work with all parameters
        audio, sr, metadata = backend.generate(
            prompt="test",
            audio_length=1.0,
            cfg_scale=7.0,
            steps=100
        )
        assert audio is not None


@pytest.mark.unit
class TestBackendOutputValidation:
    """Test that backend outputs are valid."""

    def test_audio_array_is_stereo(self):
        """Test that generated audio is stereo (2 channels)."""
        backend = MockBackend()
        audio, sr, metadata = backend.generate("test", audio_length=1.0)

        assert audio.shape[0] == 2, "Audio should be stereo (2 channels)"

    def test_audio_array_has_correct_length(self):
        """Test that generated audio has correct length."""
        backend = MockBackend()
        audio_length = 3.0

        audio, sr, metadata = backend.generate("test", audio_length=audio_length)

        expected_samples = int(audio_length * sr)
        actual_samples = audio.shape[1]

        # Allow 1% tolerance
        tolerance = expected_samples * 0.01
        assert abs(actual_samples - expected_samples) < tolerance

    def test_audio_array_in_valid_range(self):
        """Test that audio values are in valid range [-1, 1]."""
        backend = MockBackend(mode="noise")
        audio, sr, metadata = backend.generate("test", audio_length=1.0)

        assert np.all(audio >= -1.0), "Audio values should be >= -1.0"
        assert np.all(audio <= 1.0), "Audio values should be <= 1.0"

    def test_metadata_contains_required_fields(self):
        """Test that metadata contains required fields."""
        backend = MockBackend()
        audio, sr, metadata = backend.generate("test", audio_length=1.0)

        required_fields = [
            "backend",
            "prompt",
            "sample_rate",
            "channels",
            "num_samples",
            "duration_seconds"
        ]

        for field in required_fields:
            assert field in metadata, f"Metadata missing required field: {field}"
