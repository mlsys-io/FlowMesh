from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.utils import new_supervisor_command_id


class CommandType(StrEnum):
    START_WORKER = "START_WORKER"
    CREATE_WORKER = "CREATE_WORKER"  # payload: WorkerInitConfig dict
    CREATE_WORKER_ON_NODE = (
        "CREATE_WORKER_ON_NODE"  # payload: DockerWorkerConfig + gpu_count hint
    )
    GET_WORKERS = "GET_WORKERS"
    STOP_WORKER = "STOP_WORKER"
    DESTROY_WORKER = "DESTROY_WORKER"  # payload: {worker_name: str}
    DESTROY_WORKERS = "DESTROY_WORKERS"  # payload: {worker_names: [str]} or null
    START_RELAY = "START_RELAY"


class CommandMessage(BaseModel):
    command: CommandType
    command_id: str = Field(default_factory=new_supervisor_command_id)
    payload: dict[str, Any] | None = None


class CommandResponse(BaseModel):
    command_id: str
    success: bool
    message: str | None = None
    data: dict[str, Any] | None = None

    @classmethod
    def ok(
        cls, cmd: CommandMessage, data: dict[str, Any] | None = None
    ) -> "CommandResponse":
        return cls(command_id=cmd.command_id, success=True, data=data)

    @classmethod
    def error(cls, cmd: CommandMessage, message: str) -> "CommandResponse":
        return cls(command_id=cmd.command_id, success=False, message=message)


class InterruptMessage(BaseModel):
    kind: Literal["interrupt"] = "interrupt"
    task_id: str
    worker_id: str
    reason: str = "cancelled"


class StopMessage(BaseModel):
    kind: Literal["stop"] = "stop"
    task_id: str
    worker_id: str
    reason: str = "stopped"


class TaskMessage(BaseModel):
    kind: Literal["task"] = "task"
    worker_id: str
    payload: dict[str, Any]


type DispatchMessage = TaskMessage | InterruptMessage | StopMessage


__all__ = [
    "CommandMessage",
    "CommandResponse",
    "CommandType",
    "DispatchMessage",
    "TaskMessage",
    "InterruptMessage",
    "StopMessage",
]
