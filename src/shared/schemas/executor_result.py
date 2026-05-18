from pydantic import BaseModel, ConfigDict, Field

from .artifact import ArtifactContext


class BaseExecutorResult(BaseModel):
    """Common shape for every executor's result payload.

    ``extra="allow"`` lets the server round-trip subclass payloads through
    this base class without losing executor-specific fields.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    children: dict[str, "BaseExecutorResult"] = Field(
        default_factory=dict,
        description="Per-child result payloads for task merging.",
    )
    artifacts: ArtifactContext | None = Field(
        default=None,
        alias="_artifacts",
        description="Resolution context for relative artifact refs.",
    )


BaseExecutorResult.model_rebuild()
