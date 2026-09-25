from pydantic import BaseModel, Field


class WorkerCordon(BaseModel):
    node_alias: str = Field(description="Alias of the node the worker runs on.")
    alias: str = Field(description="Worker alias.")


class WorkerCordonResult(WorkerCordon):
    cordoned: bool = Field(description="Whether the worker is now cordoned.")
    changed: bool = Field(description="Whether this request changed the cordon.")
