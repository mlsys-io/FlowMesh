from pydantic import BaseModel, Field


class NodeInfo(BaseModel):
    namespace: str = Field(description="Node namespace.")
    cluster: str = Field(description="Node cluster.")
    alias: str = Field(description="Human-readable node alias.")
    version: str | None = Field(default=None, description="Node version.")
    started_at: str = Field(description="Node start timestamp.")
    tags: list[str] = Field(description="Node tags.")
    last_seen: str = Field(description="Last heartbeat timestamp.")
    max_gpu_count: int = Field(description="Total GPU count available on this node.")


__all__ = ["NodeInfo"]
