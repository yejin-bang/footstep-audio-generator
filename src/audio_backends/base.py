"""
Abstract Base Class for Audio Generation Backends

Defines the interface that all audio generation backends must implement.
This allows the pipeline to work with different generation methods
(RunPod, local GPU, cloud APIs, etc.) without code changes.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np
from pathlib import Path


class AudioBackend(ABC):
    """
    Abstract base class for audio generation backends.

    All backends must implement the generate() method to produce
    audio from text prompts.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate audio from text prompt.

        Args:
            prompt: Text description of the audio to generate
            audio_length: Duration of audio in seconds
            cfg_scale: Classifier-free guidance scale
            steps: Number of diffusion steps
            **kwargs: Backend-specific parameters

        Returns:
            Tuple of:
                - audio: NumPy array of shape [channels, samples]
                - sample_rate: Audio sample rate (Hz)
                - metadata: Dict with generation info (duration, device, etc.)
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, int, Path, Dict]:
        """
        Generate audio and save to file.

        This is a convenience method that calls generate() and saves the result.
        Subclasses can override this for custom saving behavior.

        Args:
            prompt: Text description of the audio to generate
            output_path: Where to save the audio file
            audio_length: Duration of audio in seconds
            cfg_scale: Classifier-free guidance scale
            steps: Number of diffusion steps
            **kwargs: Backend-specific parameters

        Returns:
            Tuple of:
                - audio: NumPy array of shape [channels, samples]
                - sample_rate: Audio sample rate (Hz)
                - output_path: Path object where audio was saved
                - metadata: Dict with generation info
        """
        import soundfile as sf
        from datetime import datetime

        # Generate audio
        audio, sample_rate, metadata = self.generate(
            prompt=prompt,
            audio_length=audio_length,
            cfg_scale=cfg_scale,
            steps=steps,
            **kwargs
        )

        # Prepare output path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp if file exists
        if output_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"

        # Save audio (transpose for soundfile: [samples, channels])
        sf.write(output_path, audio.T, sample_rate)

        # Update metadata
        metadata["output_path"] = str(output_path)
        metadata["prompt"] = prompt

        return audio, sample_rate, output_path, metadata

    def get_info(self) -> Dict:
        """
        Get information about this backend.

        Returns:
            Dict with backend name, version, capabilities, etc.
        """
        return {
            "name": self.__class__.__name__,
            "type": "abstract",
        }

    def __repr__(self):
        """String representation of backend."""
        info = self.get_info()
        return f"{info['name']}()"
