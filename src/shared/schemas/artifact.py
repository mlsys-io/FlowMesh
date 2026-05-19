from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    path: str = Field(description="Path relative to the task's artifacts/ dir.")


class ArtifactContext(BaseModel):
    base_dir: str = Field(description="Producing task's output directory.")
    base_url: str | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="HTTP origin (scheme://host[:port]) for upload.",
    )
