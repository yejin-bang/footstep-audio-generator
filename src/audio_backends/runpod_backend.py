"""
RunPod Backend for Audio Generation

Implements the AudioBackend interface using RunPod serverless GPU endpoints.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

from ..utils.runpod_api import RunPodClient, RunPodError
from .base import AudioBackend


class RunPodBackend(AudioBackend):
    """
    RunPod serverless GPU backend for audio generation.

    Uses RunPod's async API to generate audio on remote GPUs.
    Requires RUNPOD_API_KEY and RUNPOD_ENDPOINT_URL in environment
    or passed as arguments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        timeout: Optional[int] = None,
        poll_interval: float = 2.0
    ):
        """
        Initialize RunPod backend.

        Args:
            api_key: RunPod API key (or use RUNPOD_API_KEY env var)
            endpoint_url: RunPod endpoint URL (or use RUNPOD_ENDPOINT_URL env var)
            timeout: Request timeout in seconds (default: 300)
            poll_interval: How often to poll for results (default: 2.0s)
        """
        try:
            self.client = RunPodClient(
                api_key=api_key,
                endpoint_url=endpoint_url,
                timeout=timeout,
                poll_interval=poll_interval
            )
        except ValueError as e:
            raise ValueError(
                f"RunPod backend initialization failed: {e}\n"
                "Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_URL in .env file "
                "or pass them as arguments."
            )

    def generate(
        self,
        prompt: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate audio using RunPod serverless GPU.

        Args:
            prompt: Text description of audio to generate
            audio_length: Duration in seconds (default: 6.0)
            cfg_scale: Classifier-free guidance scale (default: 7.0)
            steps: Number of diffusion steps (default: 100)
            **kwargs: Additional parameters (unused)

        Returns:
            Tuple of (audio_array, sample_rate, metadata)
            - audio_array: Shape [channels, samples]
            - sample_rate: Integer sample rate in Hz
            - metadata: Dict with generation info
        """
        try:
            # Call RunPod API
            audio, sample_rate = self.client.generate(
                prompt=prompt,
                audio_length=audio_length,
                cfg_scale=cfg_scale,
                steps=steps
            )

            # Build metadata
            channels, samples = audio.shape
            duration = samples / sample_rate

            metadata = {
                "backend": "runpod",
                "prompt": prompt,
                "sample_rate": sample_rate,
                "channels": channels,
                "num_samples": samples,
                "duration_seconds": round(duration, 2),
                "cfg_scale": cfg_scale,
                "diffusion_steps": steps,
                "max_amplitude": round(float(np.max(np.abs(audio))), 6),
                "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
            }

            return audio, sample_rate, metadata

        except RunPodError as e:
            raise RuntimeError(f"RunPod generation failed: {e}") from e

    def get_info(self) -> Dict:
        """Get backend information."""
        return {
            "name": "RunPodBackend",
            "type": "serverless_gpu",
            "provider": "RunPod",
            "supports_local": False,
            "requires_api_key": True,
            "endpoint": getattr(self.client, 'endpoint_url', 'Not configured'),
        }

    def __repr__(self):
        """String representation."""
        return f"RunPodBackend(endpoint={self.client.endpoint_url[:50]}...)"
