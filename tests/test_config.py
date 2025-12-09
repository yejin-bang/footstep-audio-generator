"""
Unit Tests for Configuration Module

Tests centralized configuration and path management.
"""

import pytest
from pathlib import Path
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    PROJECT_ROOT,
    CONFIG_DIR,
    CAPTION_CONFIG_PATH,
    get_test_video,
    get_test_audio,
    list_test_videos,
    list_test_audios,
    validate_config,
    ensure_output_dirs,
    get_config_value
)


@pytest.mark.unit
class TestConfigPaths:
    """Test path configuration constants."""

    def test_project_root_exists(self):
        """Test that project root directory exists."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_config_dir_exists(self):
        """Test that config directory exists."""
        assert CONFIG_DIR.exists()
        assert CONFIG_DIR.is_dir()

    def test_caption_config_exists(self):
        """Test that caption config file exists."""
        assert CAPTION_CONFIG_PATH.exists()
        assert CAPTION_CONFIG_PATH.is_file()
        assert CAPTION_CONFIG_PATH.suffix == ".json"


@pytest.mark.unit
class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_list_test_videos(self):
        """Test listing test videos."""
        videos = list_test_videos()
        assert isinstance(videos, list)
        # May be empty if test_videos/ doesn't exist
        for video in videos:
            assert video.endswith('.mp4')

    def test_list_test_audios(self):
        """Test listing test audio files."""
        audios = list_test_audios()
        assert isinstance(audios, list)
        # May be empty if test_audios/ doesn't exist
        for audio in audios:
            assert audio.endswith('.wav')

    def test_get_config_value(self):
        """Test getting configuration values from environment."""
        # Test with default
        value = get_config_value("NONEXISTENT_KEY", "default_value")
        assert value == "default_value"

        # Test with actual env var
        os.environ["TEST_CONFIG_KEY"] = "test_value"
        value = get_config_value("TEST_CONFIG_KEY")
        assert value == "test_value"
        del os.environ["TEST_CONFIG_KEY"]

    def test_ensure_output_dirs(self):
        """Test creating output directories."""
        # Should not raise any errors
        ensure_output_dirs()

        # Verify directories were created
        from src.utils.config import PIPELINE_OUTPUTS_DIR, GENERATED_OUTPUTS_DIR, LOGS_DIR
        assert PIPELINE_OUTPUTS_DIR.exists()
        assert GENERATED_OUTPUTS_DIR.exists()
        assert LOGS_DIR.exists()


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_config_structure(self):
        """Test that validate_config returns proper structure."""
        result = validate_config()

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result

        assert isinstance(result['valid'], bool)
        assert isinstance(result['errors'], list)
        assert isinstance(result['warnings'], list)

    def test_validate_config_caption_config(self):
        """Test that validation checks for caption config."""
        result = validate_config()

        # Caption config should exist (it's required)
        if not result['valid']:
            # If invalid, should have error about caption config
            error_messages = " ".join(result['errors'])
            assert "caption" in error_messages.lower() or "config" in error_messages.lower()


@pytest.mark.unit
class TestGetTestResources:
    """Test getting test resource files."""

    def test_get_test_video_raises_on_missing(self):
        """Test that get_test_video raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            get_test_video("nonexistent_video.mp4")

    def test_get_test_audio_raises_on_missing(self):
        """Test that get_test_audio raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            get_test_audio("nonexistent_audio.wav")

    def test_get_test_video_returns_path(self):
        """Test that get_test_video returns valid Path object if file exists."""
        videos = list_test_videos()
        if videos:
            # If we have test videos, test with the first one
            video_path = get_test_video(videos[0])
            assert isinstance(video_path, Path)
            assert video_path.exists()
            assert video_path.suffix == ".mp4"

    def test_get_test_audio_returns_path(self):
        """Test that get_test_audio returns valid Path object if file exists."""
        audios = list_test_audios()
        if audios:
            # If we have test audios, test with the first one
            audio_path = get_test_audio(audios[0])
            assert isinstance(audio_path, Path)
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"
