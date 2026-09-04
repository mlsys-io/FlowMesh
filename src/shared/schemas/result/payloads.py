"""Typed nested payload models describing the exact shape each executor emits
inside its result fields (items, usage, cost estimates, ...)."""

from typing import Any

from pydantic import ConfigDict, Field, JsonValue

from ..artifact import ArtifactRef
from ._base import DropNoneModel, StrictModel


class GenerationUsage(StrictModel):
    """Token/latency accounting for text-generation inference."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float


class EmbeddingUsage(StrictModel):
    """Token/latency accounting for embedding inference."""

    prompt_tokens: int
    total_tokens: int
    num_requests: int
    latency_sec: float
    embedding_dim: int


class InferenceItem(StrictModel):
    """One prompt's generation output.

    ``output`` is polymorphic: a plain string, a structured JSON value when a
    template schema is applied, or a list of grouped outputs (table / grouped-
    image modes). ``metadata`` is open dataset/user passthrough.
    """

    index: int
    prompt: str
    output: JsonValue
    finish_reason: str | list[str | None] | None
    metadata: dict[str, Any] | None = None


class OmniImageItem(StrictModel):
    """One text-to-image generation."""

    index: int
    prompt: str
    image: ArtifactRef


class OmniSpeechItem(StrictModel):
    """One text-to-speech generation."""

    index: int
    text: str
    audio: ArtifactRef


class OmniAudioItem(StrictModel):
    """One text-to-audio (BGM) waveform."""

    index: int
    prompt_index: int
    waveform_index: int
    prompt: str
    audio: ArtifactRef


class OmniGeneralItem(StrictModel):
    """One text-to-general (narration) segment."""

    index: int
    request_id: str
    prompt: str
    audio: ArtifactRef
    text: str | None = None


class CostEstimates(StrictModel):
    """Aggregated query cost/row estimates for data profiling."""

    ok: bool
    num_queries: int
    avg_estimated_cost: float
    min_estimated_cost: float
    max_estimated_cost: float
    avg_estimated_rows: float
    min_estimated_rows: int
    max_estimated_rows: int


class DataRetrievalItem(DropNoneModel):
    """One retrieved row/object.

    Shapes differ across the SQL, S3, and Lumid (sql/agent) connectors, and
    the Lumid fields (``access_chain``, token/step metrics) pass through a
    remote contract verbatim. Declared fields are typed; ``extra="allow"``
    keeps the item robust to connector-specific additions. ``access_chain``
    is an opaque provenance object and stays untyped.
    """

    model_config = ConfigDict(extra="allow")

    index: int | None = None
    query: str | None = None
    description: str | None = None
    params: Any = None
    table: dict[str, str] | None = None
    rows: int | None = None
    keys: list[str] | None = None
    content: list[Any] | None = None
    run_id: str | None = None
    access_chain: Any = None
    materialized_uri: str | None = None
    size_bytes: int | None = None
    transcript_url: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    steps_taken: int | None = None
    replay_latency_ms: int | float | None = None


class AgentItem(StrictModel):
    """One agent task's final output."""

    index: int
    output: str
    finish_reason: str


class AgentUsage(StrictModel):
    """Agent execution accounting."""

    execution_time_sec: float
    num_requests: int
    agent_config: str


class AgentBatchSummary(StrictModel):
    """Per-batch agent completion counts."""

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0


class AgentMetadata(StrictModel):
    """Agent run metadata (single, batch, or error variant)."""

    task: str | None = None
    tasks_count: int | None = None
    execution_log: list[str] = Field(default_factory=list)
    error: str | None = None
    batch_summary: AgentBatchSummary | None = None


class RagQdrant(StrictModel):
    """Qdrant collection the RAG query ran against."""

    collection: str
    url: str


class RagEmbedding(StrictModel):
    """Embedding model used for the RAG query."""

    model: str


class RagSearch(StrictModel):
    """RAG search parameters."""

    top_k: int


class RagUsage(StrictModel):
    """RAG query accounting."""

    latency_sec: float
    num_queries: int
    total_results: int


class RagHit(StrictModel):
    """One Qdrant search hit. ``payload`` is the arbitrary stored document."""

    id: int | str | None = None
    score: float | None = None
    payload: dict[str, Any] | None = None


class RagQuery(StrictModel):
    """Hits for one RAG query."""

    index: int
    query: str
    items: list[RagHit] = Field(default_factory=list)


class EchoItem(StrictModel):
    """One echoed value."""

    output: JsonValue = None
