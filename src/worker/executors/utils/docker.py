"""Helpers for connecting to the host Docker daemon.

Centralizes the single way the worker reaches Docker so the SSH executor and the
startup capability probe share one connection path and one failure definition.
"""

from typing import TYPE_CHECKING, Any

try:
    import docker
    from docker import DockerClient

    _HAS_DOCKER = True
except Exception:
    _HAS_DOCKER = False
    if TYPE_CHECKING:
        import docker
        from docker import DockerClient
    else:
        docker = None
        DockerClient = Any


class DockerUnavailableError(RuntimeError):
    """The host Docker daemon could not be reached.

    Raised when the SDK is missing, the socket is not mounted, the daemon is
    down, or the worker lacks permission to use the socket.
    """


def docker_client() -> "DockerClient":
    """Connect to the host Docker daemon.

    Raises ``DockerUnavailableError`` when a connection cannot be established.
    """
    if not _HAS_DOCKER:
        raise DockerUnavailableError("docker SDK is not installed")
    try:
        return docker.from_env()
    except Exception as exc:
        raise DockerUnavailableError(str(exc)) from exc


def docker_available() -> bool:
    """Return whether the host Docker daemon is reachable."""
    try:
        docker_client().ping()
        return True
    except Exception:
        return False
