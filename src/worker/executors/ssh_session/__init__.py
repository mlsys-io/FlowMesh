"""SSH session backends and the configuration they share."""

import logging

from shared.schemas.worker import SSHBackendName
from worker.config import WorkerConfig

from .backends.docker import DockerSessionBackend
from .backends.process import ProcessSessionBackend
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

logger = logging.getLogger(__name__)

BACKENDS: dict[SSHBackendName, type[SSHSessionBackend]] = {
    DockerSessionBackend.name: DockerSessionBackend,
    ProcessSessionBackend.name: ProcessSessionBackend,
}


def _resolve_auto(config: WorkerConfig) -> type[SSHSessionBackend] | None:
    """Pick a backend for ``auto``, preferring the isolation Docker gives."""
    for backend_cls in (DockerSessionBackend, ProcessSessionBackend):
        if backend_cls.is_available(config):
            return backend_cls
    return None


def select_backend_cls(config: WorkerConfig) -> type[SSHSessionBackend] | None:
    """Resolve the session backend this worker should use, if any."""
    raw = (config.ssh_session_backend or SSHBackendName.AUTO).strip().lower()
    try:
        requested = SSHBackendName(raw)
    except ValueError:
        logger.warning(
            "Unknown SSH_SESSION_BACKEND %r; expected one of %s",
            raw,
            ", ".join(sorted(SSHBackendName)),
        )
        return None
    if requested is SSHBackendName.AUTO:
        return _resolve_auto(config)
    backend_cls = BACKENDS[requested]
    return backend_cls if backend_cls.is_available(config) else None


__all__ = [
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
