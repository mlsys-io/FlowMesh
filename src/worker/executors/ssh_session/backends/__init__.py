"""Session backends: the sandboxes an SSH session can run in."""

from .docker import DockerSessionBackend
from .process import ProcessSessionBackend

__all__ = ["DockerSessionBackend", "ProcessSessionBackend"]
