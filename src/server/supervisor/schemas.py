from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

from shared.schemas.worker import SSHLimits, WorkerCapabilities

from .. import env
from ..schemas.node import WorkerHardware


class WorkerStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class WorkerInfo(BaseModel):
    id: Annotated[str | None, Field(description="Worker ID")]
    name: Annotated[str, Field(description="Worker name")]
    namespace: str = Field(default=env.NODE_NAMESPACE, description="Worker namespace")
    cluster: str = Field(default=env.NODE_CLUSTER, description="Worker cluster")
    node_alias: str = Field(default=env.NODE_ALIAS, description="Node alias")
    provider: Annotated[str, Field(description="Worker provider")]
    status: Annotated[WorkerStatus, Field(description="Current worker status")]
    hardware: Annotated[
        WorkerHardware | None, Field(default=None, description="Hardware metadata")
    ]
    ssh_limits: Annotated[
        SSHLimits | None,
        Field(
            default=None,
            description="Configured ceiling on SSH session resources.",
        ),
    ] = None
    capabilities: WorkerCapabilities = Field(
        default_factory=WorkerCapabilities,
        description="Task capabilities the worker advertises.",
    )


__all__ = ["WorkerCapabilities", "WorkerHardware", "WorkerInfo", "WorkerStatus"]
