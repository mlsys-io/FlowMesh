"""OpenTelemetry tracing wiring for worker executors.

Sets up a single process-wide ``TracerProvider`` with a JSONL exporter that
appends ``ReadableSpan.to_json()`` to ``<out_dir>/logs/spans.jsonl``
for whichever task is currently executing. The current path is held in a
module-level slot updated by ``_task_span`` on enter / exit; the worker is
single-threaded for executor work so there's no contention.

The ``trace_id`` is pinned to the workflow id via a custom ``IdGenerator``
that reads the active workflow id from a ``ContextVar`` populated by
``_task_span``. Sub-spans inherit the trace id from the OTel parent context
automatically.
"""

import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

from shared.governance.spans import FlowMeshSpanKind
from shared.utils.ids import PREFIX_WORKFLOW

_HEX_ONLY = re.compile(r"[^0-9a-f]")
_TRACER_NAME = "flowmesh.worker"
_SERVICE_NAME = "flowmesh-worker"

_workflow_id_var: ContextVar[str | None] = ContextVar(
    "flowmesh_workflow_id", default=None
)
_lock = threading.Lock()
_current_spans_path: Path | None = None


def workflow_to_trace_id_int(workflow_id: str) -> int:
    """Stable 128-bit trace id derived from the workflow id.

    Strips the ``wfl-`` prefix before hex extraction so the prefix's ``f``
    doesn't shift the bit pattern.
    """
    body = workflow_id.lower().removeprefix(f"{PREFIX_WORKFLOW}-")
    hex_only = _HEX_ONLY.sub("", body)
    if not hex_only:
        return 0
    return int(hex_only.zfill(32)[:32], 16)


class _FlowMeshIdGenerator(IdGenerator):
    """Pin trace_id to the active workflow id; random span_ids."""

    def __init__(self) -> None:
        self._fallback = RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._fallback.generate_span_id()

    def generate_trace_id(self) -> int:
        workflow_id = _workflow_id_var.get()
        if workflow_id:
            value = workflow_to_trace_id_int(workflow_id)
            if value != 0:
                return value
        return self._fallback.generate_trace_id()


class _JSONLSpanExporter(SpanExporter):
    """Append each completed span to the active task's spans.jsonl file."""

    def __init__(self, path_provider: Callable[[], Path | None]) -> None:
        self._path_provider = path_provider

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        path = self._path_provider()
        if path is None:
            return SpanExportResult.SUCCESS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for span in spans:
                fh.write(span.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _resolve_path() -> Path | None:
    return _current_spans_path


_PROVIDER_INITIALIZED = False


def _ensure_tracer_provider() -> None:
    global _PROVIDER_INITIALIZED
    if _PROVIDER_INITIALIZED:
        return
    with _lock:
        if _PROVIDER_INITIALIZED:
            return
        provider = TracerProvider(
            resource=Resource.create({"service.name": _SERVICE_NAME}),
            id_generator=_FlowMeshIdGenerator(),
        )
        provider.add_span_processor(
            SimpleSpanProcessor(_JSONLSpanExporter(_resolve_path))
        )
        trace.set_tracer_provider(provider)
        _PROVIDER_INITIALIZED = True


def get_tracer():
    _ensure_tracer_provider()
    return trace.get_tracer(_TRACER_NAME)


def _set_active_spans_path(path: Path | None) -> None:
    global _current_spans_path
    _current_spans_path = path


@contextmanager
def task_trace_context(workflow_id: str, spans_path: Path) -> Iterator[None]:
    """Bind trace_id and span exporter destination for the duration of a task.

    Pins trace_id derivation to ``workflow_id`` and routes the JSONL exporter
    to ``spans_path`` while the block is active. Restores the previous state
    on exit.
    """
    token = _workflow_id_var.set(workflow_id)
    _set_active_spans_path(spans_path)
    try:
        yield
    finally:
        _set_active_spans_path(None)
        _workflow_id_var.reset(token)


def attributes_with_kind(
    flowmesh_kind: FlowMeshSpanKind,
    *,
    data_id: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {"flowmesh.kind": flowmesh_kind.value}
    if data_id is not None:
        attrs["data_id"] = data_id
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            attrs[key] = value
    return attrs
