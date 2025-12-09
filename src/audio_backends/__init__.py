"""
Audio Generation Backends

Pluggable backend system for audio generation. Supports multiple
generation methods (RunPod, local GPU, cloud APIs, etc.)

Usage:
    from src.audio_backends import get_backend

    # Get RunPod backend
    backend = get_backend("runpod")
    audio, sr, metadata = backend.generate(prompt="boots on marble")

    # Users can implement their own backends by extending AudioBackend
"""

from .base import AudioBackend
from .runpod_backend import RunPodBackend
from .mock_backend import MockBackend

# Backend registry
_BACKENDS = {
    "runpod": RunPodBackend,
    "mock": MockBackend,
    # Future backends can be registered here:
    # "local": LocalGPUBackend,
    # "huggingface": HuggingFaceBackend,
    # "replicate": ReplicateBackend,
}


def get_backend(backend_name: str, **kwargs) -> AudioBackend:
    """
    Factory function to get an audio generation backend.

    Args:
        backend_name: Name of backend ("runpod", "mock", etc.)
        **kwargs: Backend-specific configuration

    Returns:
        Initialized AudioBackend instance

    Raises:
        ValueError: If backend_name is not registered

    Example:
        >>> backend = get_backend("runpod", api_key="...", endpoint_url="...")
        >>> audio, sr, metadata = backend.generate("footsteps on wood")
    """
    if backend_name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: {available}"
        )

    backend_class = _BACKENDS[backend_name]
    return backend_class(**kwargs)


def list_backends():
    """List all available backends."""
    return list(_BACKENDS.keys())


def register_backend(name: str, backend_class: type):
    """
    Register a custom backend.

    Args:
        name: Backend name
        backend_class: Class inheriting from AudioBackend

    Example:
        >>> class MyCustomBackend(AudioBackend):
        ...     def generate(self, prompt, **kwargs):
        ...         # Custom implementation
        ...         pass
        >>>
        >>> register_backend("custom", MyCustomBackend)
        >>> backend = get_backend("custom")
    """
    if not issubclass(backend_class, AudioBackend):
        raise TypeError(f"{backend_class} must inherit from AudioBackend")

    _BACKENDS[name] = backend_class


__all__ = [
    "AudioBackend",
    "RunPodBackend",
    "MockBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
