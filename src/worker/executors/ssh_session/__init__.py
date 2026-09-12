"""SSH session backends and the configuration they share."""

import logging

from worker.config import WorkerConfig

from .base import (
    LOOPBACK_RELAY_HOST,
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    resolve_tailnet_address,
)
from .config import (
    ResolvedSSHInput,
    SSHConfig,
    SSHOutputConfig,
    normalize_mount_path,
    reserve_mount_path,
)
from .docker_backend import DockerSessionBackend
from .process_backend import ProcessSessionBackend

logger = logging.getLogger(__name__)

AUTO_BACKEND = "auto"

BACKENDS: dict[str, type[SSHSessionBackend]] = {
    DockerSessionBackend.name: DockerSessionBackend,
    ProcessSessionBackend.name: ProcessSessionBackend,
}


def select_backend_cls(config: WorkerConfig) -> type[SSHSessionBackend] | None:
    """Resolve the session backend this worker should use, if any.

    ``auto`` resolves to Docker alone: a worker image that happens to ship
    ``sshd`` must not silently downgrade to running sessions in its own
    namespace, so process mode is opt-in.
    """
    requested = (config.ssh_session_backend or AUTO_BACKEND).strip().lower()
    if requested == AUTO_BACKEND:
        backend_cls: type[SSHSessionBackend] = DockerSessionBackend
    else:
        selected = BACKENDS.get(requested)
        if selected is None:
            logger.warning(
                "Unknown SSH_SESSION_BACKEND %r; expected one of %s or %r",
                requested,
                ", ".join(sorted(BACKENDS)),
                AUTO_BACKEND,
            )
            return None
        backend_cls = selected
    return backend_cls if backend_cls.is_available(config) else None


__all__ = [
    "AUTO_BACKEND",
    "BACKENDS",
    "LOOPBACK_RELAY_HOST",
    "DockerSessionBackend",
    "ProcessSessionBackend",
    "ResolvedSSHInput",
    "SSHConfig",
    "SSHOutputConfig",
    "SSHSession",
    "SSHSessionBackend",
    "SessionRequest",
    "count_established_connections",
    "normalize_mount_path",
    "reserve_mount_path",
    "resolve_tailnet_address",
    "select_backend_cls",
]
