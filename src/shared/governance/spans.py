"""Pydantic models that parse OTel JSON spans into Python objects.

Producers (the worker mixin) write spans via ``ReadableSpan.to_json()``.
Consumers (the analyzer, server endpoint, SDK) parse with
:class:`Span.parse_otel_json` and work with strongly-typed access.
"""

import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlowMeshSpanKind(StrEnum):
    """Worker-side classification carried in ``attributes["flowmesh.kind"]``."""

    COMPUTE = "compute"
    NETWORK = "network"
    MARKER = "marker"


def _strip_hex_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    return value[2:] if value.startswith("0x") else value


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Span(BaseModel):
    """Parsed OTel JSON span row.

    The OTel SDK serializes ids as ``0x<hex>`` strings and timestamps as
    ISO8601. We strip the ``0x`` prefix on parse and expose ``start_time`` /
    ``end_time`` as :class:`datetime` for arithmetic.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: datetime
    end_time: datetime
    attributes: dict[str, Any] = Field(default_factory=dict)
    status_code: str = "UNSET"
    status_message: str | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def data_id(self) -> str | None:
        value = self.attributes.get("data_id")
        return str(value) if value else None

    @property
    def flowmesh_kind(self) -> FlowMeshSpanKind | None:
        raw = self.attributes.get("flowmesh.kind")
        if not isinstance(raw, str):
            return None
        try:
            return FlowMeshSpanKind(raw)
        except ValueError:
            return None

    @property
    def batch_id(self) -> str | None:
        value = self.attributes.get("batch_id")
        return str(value) if value else None

    @classmethod
    def parse_otel_json(cls, raw: str | dict[str, Any]) -> "Span":
        payload = json.loads(raw) if isinstance(raw, str) else raw
        ctx = payload.get("context") or {}
        status = payload.get("status") or {}
        raw_attrs = payload.get("attributes") or {}
        return cls(
            name=str(payload.get("name") or ""),
            trace_id=_strip_hex_prefix(ctx.get("trace_id")) or "",
            span_id=_strip_hex_prefix(ctx.get("span_id")) or "",
            parent_span_id=_strip_hex_prefix(payload.get("parent_id")),
            start_time=_parse_iso(payload.get("start_time"))
            or datetime.fromtimestamp(0),
            end_time=_parse_iso(payload.get("end_time")) or datetime.fromtimestamp(0),
            attributes=raw_attrs.copy(),
            status_code=str(status.get("status_code") or "UNSET"),
            status_message=(
                str(status.get("description")) if status.get("description") else None
            ),
        )
