from enum import StrEnum

from pydantic import BaseModel, Field


class WorkerStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"


class SSHLimits(BaseModel):
    """Per-worker ceiling for resources accessible by SSH session containers.

    Populated from the worker's ``SSH_MAX_*`` configuration. Used by the dispatcher to
    filter workers for SSH tasks and by the worker at runtime to clamp the spawned
    container's cgroup limits.
    """

    max_cpu_cores: float | None = Field(
        default=None, description="Maximum CPU cores accessible to an SSH session."
    )
    max_memory_bytes: int | None = Field(
        default=None,
        description="Maximum memory in bytes accessible to an SSH session.",
    )
    max_pids: int | None = Field(
        default=None, description="Maximum number of PIDs inside an SSH session."
    )


class WorkerCapabilities(BaseModel):
    """Task capabilities a worker advertises to the dispatcher.

    Each flag reflects what the worker process can actually service, as opposed
    to what its node was configured to allow. A capability left at its default
    means the worker advertises that it cannot service the corresponding task
    kind, so the dispatcher excludes it.
    """

    ssh: bool = Field(
        default=False,
        description="Whether the worker can run SSH session tasks (Docker reachable).",
    )


__all__ = ["SSHLimits", "WorkerCapabilities", "WorkerStatus"]
