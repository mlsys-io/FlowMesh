from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from shared.tasks.task_type import TaskType


class WorkerStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    # Registered and alive, but temporarily not dispatchable. The dispatcher
    # only picks IDLE workers.
    UNAVAILABLE = "UNAVAILABLE"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        return cls.UNKNOWN


class SSHBackendName(StrEnum):
    """Sandbox an SSH session runs in, as named by ``SSH_SESSION_BACKEND``."""

    AUTO = "auto"
    DOCKER = "docker"
    PROCESS = "process"


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
    """Task capabilities a worker advertises to the dispatcher."""

    supported_task_types: frozenset[TaskType] = Field(
        default_factory=frozenset,
        description="Types of tasks this worker can service.",
    )


__all__ = ["SSHBackendName", "SSHLimits", "WorkerCapabilities", "WorkerStatus"]
