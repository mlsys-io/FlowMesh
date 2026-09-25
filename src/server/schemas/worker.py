from pydantic import BaseModel, ConfigDict, Field


class WorkerCordon(BaseModel):
    node_alias: str = Field(description="Alias of the node the worker runs on.")
    alias: str = Field(description="Worker alias.")


class WorkerCordonById(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(description="Worker identifier.")


class WorkerCordonByAlias(WorkerCordon):
    model_config = ConfigDict(extra="forbid")

    node_alias: str = Field(
        min_length=1, description="Alias of the node the worker runs on."
    )
    alias: str = Field(min_length=1, description="Worker alias.")


WorkerCordonRequest = WorkerCordonById | WorkerCordonByAlias


class WorkerCordonResult(WorkerCordon):
    cordoned: bool = Field(description="Whether the worker is now cordoned.")
    changed: bool = Field(description="Whether this request changed the cordon.")
    worker_ids: list[str] = Field(
        description="Live workers currently registered under this key."
    )
