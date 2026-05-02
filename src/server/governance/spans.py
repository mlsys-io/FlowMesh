"""Server-side parser for OTel-shape span rows in ``spans.jsonl``.

Producers (workers) emit one ``ReadableSpan.to_json()`` row per line; the
analyzer reads them as :class:`Span` instances. The shared wire-contract enum
:class:`shared.governance.spans.FlowMeshSpanKind` lives in
``src/shared/governance/spans.py`` because workers also need it.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from shared.governance.spans import FlowMeshSpanKind


def _strip_hex_prefix(value: Any) -> Any:
    """Trim the leading ``0x`` from OTel hex ids; pass non-strings through."""
    if isinstance(value, str) and value.startswith("0x"):
        return value[2:]
    return value


HexId = Annotated[str, BeforeValidator(_strip_hex_prefix)]
OptionalHexId = Annotated[str | None, BeforeValidator(_strip_hex_prefix)]


class SpanContext(BaseModel):
    """The ``context`` sub-object of ``ReadableSpan.to_json()``."""

    model_config = ConfigDict(extra="allow")
    trace_id: HexId
    span_id: HexId


class SpanStatus(BaseModel):
    """The ``status`` sub-object of ``ReadableSpan.to_json()``."""

    model_config = ConfigDict(extra="allow")
    status_code: str = "UNSET"
    description: str | None = None


class SpanAttributes(BaseModel):
    """FlowMesh-required span attributes; arbitrary extras are preserved."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    data_id: str | None = None
    batch_id: str | None = None
    flowmesh_kind: FlowMeshSpanKind | None = Field(default=None, alias="flowmesh.kind")


class Span(BaseModel):
    """Parsed OTel JSON span row; ids stripped of ``0x``, times as ``datetime``."""

    model_config = ConfigDict(extra="allow")

    name: str
    context: SpanContext
    parent_id: OptionalHexId = None
    start_time: datetime
    end_time: datetime
    status: SpanStatus = Field(default_factory=SpanStatus)
    attributes: SpanAttributes = Field(default_factory=SpanAttributes)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @classmethod
    def parse_otel_json(cls, raw: str | dict[str, Any]) -> "Span":
        if isinstance(raw, str):
            return cls.model_validate_json(raw)
        return cls.model_validate(raw)
