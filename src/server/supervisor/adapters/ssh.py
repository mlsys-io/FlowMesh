"""SSH session defaults a supervisor injects into the workers it creates."""

from pydantic import BaseModel

from shared.schemas.worker import SSHLimits
from shared.utils import parse_mem_to_bytes

from ... import env
from .utils import to_env_str


class SSHConfig(BaseModel):
    default_image: str | None = env.SSH_DEFAULT_IMAGE
    """Default container image for SSH sessions"""
    default_user: str | None = env.SSH_DEFAULT_USER
    """Default SSH username"""
    default_ttl_sec: float | None = env.SSH_DEFAULT_TTL_SEC
    """Default session TTL in seconds"""
    default_idle_sec: float | None = env.SSH_DEFAULT_IDLE_SEC
    """Default idle timeout in seconds"""
    max_ttl_sec: float | None = env.SSH_MAX_TTL_SEC
    """Maximum allowed TTL in seconds"""
    poll_interval_sec: float | None = env.SSH_POLL_INTERVAL_SEC
    """Session status poll interval in seconds"""
    stop_timeout_sec: float | None = env.SSH_STOP_TIMEOUT_SEC
    """Seconds to wait when stopping a session"""
    max_cpu: float | None = env.SSH_MAX_CPU
    """Maximum CPU cores accessible to an SSH session container"""
    max_memory: str | None = env.SSH_MAX_MEMORY
    """Maximum memory accessible to an SSH session container (e.g. "8Gi")"""
    max_pids: int | None = env.SSH_MAX_PIDS
    """Maximum number of PIDs inside an SSH session container"""
    enable_gpu_limit: bool = env.ENABLE_SSH_GPU_LIMIT
    """Whether to apply requested GPU limits to SSH sessions.

    If false, SSH sessions are allocated all available GPUs regardless of
    their resource requests."""
    session_backend: str | None = env.SSH_SESSION_BACKEND
    """Session backend the worker should use (``auto``, ``docker``, ``process``).

    ``process`` is for workers that are themselves the rented machine and have
    no Docker socket; it runs one session per worker."""
    relay_host: str | None = env.SSH_RELAY_HOST
    """Address at which the worker's session ports are reachable.

    Only needed when the supervisor that dials the relay uplink does not share
    a host with the worker, and the worker cannot discover a routable address
    for itself."""

    def to_env(self) -> dict[str, str]:
        """Return env vars to inject into the worker container."""
        mapping = {
            "SSH_DEFAULT_IMAGE": self.default_image,
            "SSH_DEFAULT_USER": self.default_user,
            "SSH_DEFAULT_TTL_SEC": self.default_ttl_sec,
            "SSH_DEFAULT_IDLE_SEC": self.default_idle_sec,
            "SSH_MAX_TTL_SEC": self.max_ttl_sec,
            "SSH_POLL_INTERVAL_SEC": self.poll_interval_sec,
            "SSH_STOP_TIMEOUT_SEC": self.stop_timeout_sec,
            "SSH_MAX_CPU": self.max_cpu,
            "SSH_MAX_MEMORY": self.max_memory,
            "SSH_MAX_PIDS": self.max_pids,
            "ENABLE_SSH_GPU_LIMIT": self.enable_gpu_limit,
            "SSH_SESSION_BACKEND": self.session_backend,
            "SSH_RELAY_HOST": self.relay_host,
        }
        return {k: to_env_str(v) for k, v in mapping.items() if v is not None}

    def to_limits(self) -> SSHLimits | None:
        """Project the admin cap into a wire-ready ``SSHLimits``."""
        memory_bytes: int | None = None
        if self.max_memory is not None:
            memory_bytes = parse_mem_to_bytes(self.max_memory)
            if memory_bytes is None:
                raise ValueError(
                    f"SSH_MAX_MEMORY value {self.max_memory!r} is not a valid "
                    "memory string (e.g. '8Gi', '512Mi', or a byte count)"
                )
        if self.max_cpu is None and memory_bytes is None and self.max_pids is None:
            return None
        return SSHLimits(
            max_cpu_cores=self.max_cpu,
            max_memory_bytes=memory_bytes,
            max_pids=self.max_pids,
        )
