"""
Mock Backend for Testing

Generates synthetic audio for testing the pipeline without requiring
a GPU or API access.
"""

import numpy as np
from typing import Tuple, Dict
import time
from .base import AudioBackend


class MockBackend(AudioBackend):
    """
    Mock audio generation backend for testing.

    Generates synthetic audio (sine waves, noise, or silence) instead of
    using a real model.
    """

    def __init__(self, mode: str = "footsteps", sample_rate: int = 44100):
        """
        Initialize mock backend.

        Args:
            mode: Generation mode ("sine", "noise", "silence", "footsteps")
            sample_rate: Output sample rate (default: 44100)
        """
        valid_modes = ["sine", "noise", "silence", "footsteps"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {valid_modes}")

        self.mode = mode
        self.sample_rate = sample_rate

    def generate(
        self,
        prompt: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate mock audio.

        Args:
            prompt: Text description (ignored, for compatibility)
            audio_length: Duration in seconds
            cfg_scale: Ignored (for compatibility)
            steps: Ignored (for compatibility)

        Returns:
            Tuple of (audio_array, sample_rate, metadata)
        """
        print(f"🧪 Mock Backend: Generating {audio_length}s of '{self.mode}' audio")

        # Simulate generation time
        time.sleep(0.5)

        # Calculate number of samples
        num_samples = int(audio_length * self.sample_rate)

        # Generate audio based on mode
        if self.mode == "sine":
            audio = self._generate_sine(num_samples)
        elif self.mode == "noise":
            audio = self._generate_noise(num_samples)
        elif self.mode == "silence":
            audio = self._generate_silence(num_samples)
        elif self.mode == "footsteps":
            audio = self._generate_footsteps(num_samples, audio_length)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Build metadata
        metadata = {
            "backend": "mock",
            "mode": self.mode,
            "prompt": prompt,
            "sample_rate": self.sample_rate,
            "channels": 2,
            "num_samples": num_samples,
            "duration_seconds": round(audio_length, 2),
            "cfg_scale": cfg_scale,
            "diffusion_steps": steps,
            "max_amplitude": round(float(np.max(np.abs(audio))), 6),
            "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
            "note": "This is synthetic test audio, not real generation"
        }

        return audio, self.sample_rate, metadata

    def _generate_sine(self, num_samples: int) -> np.ndarray:
        """Generate stereo sine wave (440 Hz)."""
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)
        mono = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        stereo = np.stack([mono, mono])  # Duplicate to stereo
        return stereo.astype(np.float32)

    def _generate_noise(self, num_samples: int) -> np.ndarray:
        """Generate stereo white noise."""
        stereo = np.random.randn(2, num_samples) * 0.1
        return stereo.astype(np.float32)

    def _generate_silence(self, num_samples: int) -> np.ndarray:
        """Generate silence."""
        stereo = np.zeros((2, num_samples))
        return stereo.astype(np.float32)

    def _generate_footsteps(self, num_samples: int, duration: float) -> np.ndarray:
        """
        Generate synthetic footstep-like audio.
        Creates a series of short noise bursts that mimic footsteps.
        """
        audio = np.zeros((2, num_samples))

        # Generate ~10 footsteps evenly spaced
        num_steps = int(duration * 1.5)  # ~1.5 steps per second
        step_interval = num_samples // num_steps

        for i in range(num_steps):
            # Position of this footstep
            pos = i * step_interval

            # Generate short noise burst (50ms)
            burst_length = int(0.05 * self.sample_rate)
            if pos + burst_length > num_samples:
                burst_length = num_samples - pos

            # Create attack-decay envelope
            envelope = np.linspace(1.0, 0.0, burst_length) ** 2
            burst = np.random.randn(2, burst_length) * envelope * 0.3

            # Add to audio
            audio[:, pos:pos+burst_length] += burst

        return audio.astype(np.float32)

    def get_info(self) -> Dict:
        """Get backend information."""
        return {
            "name": "MockBackend",
            "type": "synthetic",
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "supports_local": True,
            "requires_api_key": False,
            "note": "For testing only - does not use real model",
        }

    def __repr__(self):
        """String representation."""
        return f"MockBackend(mode='{self.mode}', sr={self.sample_rate})"
