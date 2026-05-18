"""Worker mixin: OTel span emission + asset/lineage JSONL row writes."""

import contextvars
import json
import logging
import threading
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry.trace import Span as OTelSpan

from shared.schemas.executor_result import BaseExecutorResult
from shared.schemas.governance import (
    READY_SPAN_NAME,
    TASK_SPAN_NAME,
    SpanType,
)
from shared.tasks.specs import TaskSpecStrictBase
from shared.utils.time import now_iso

from ..base_executor import ExecutionError
from ._otel import attributes_with_type, get_tracer, task_trace_context

logger = logging.getLogger(__name__)


class GovernanceMixin:
    """OTel span emission + asset / lineage JSONL row writes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_id: str | None = None
        self._task_out_dir: Path | None = None
        self._current_batch_id: str | None = None
        self._task_owner_id: str = ""
        self._event_lock = threading.Lock()
        self.io_executor = ThreadPoolExecutor(max_workers=32)

    def _submit_in_context(self, fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        """``io_executor.submit`` that carries the caller's ContextVars across."""
        ctx = contextvars.copy_context()
        return self.io_executor.submit(ctx.run, fn, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Span emission — context managers driven by the OTel SDK            #
    # ------------------------------------------------------------------ #
    @contextmanager
    def _task_span(
        self,
        task_id: str,
        workflow_id: str,
        out_dir: Path,
        *,
        owner_id: str = "",
    ) -> Iterator[OTelSpan]:
        """Root span for a task — wraps the executor's ``run()`` body."""
        self._task_id = task_id
        self._current_batch_id = task_id
        self._task_out_dir = Path(out_dir)
        self._task_owner_id = owner_id
        spans_path = self._lineage_dir() / "spans.jsonl"
        spans_path.parent.mkdir(parents=True, exist_ok=True)
        with task_trace_context(workflow_id, spans_path):
            with get_tracer().start_as_current_span(
                TASK_SPAN_NAME,
                attributes=attributes_with_type(
                    SpanType.COMPUTE,
                    data_id=task_id,
                    extra={
                        "batch_id": task_id,
                        "workflow_id": workflow_id,
                        "user_id": owner_id,
                        "executor.name": getattr(self, "name", None),
                    },
                ),
            ) as span:
                yield span

    @contextmanager
    def _span(
        self,
        name: str,
        *,
        span_type: SpanType = SpanType.COMPUTE,
        data_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[OTelSpan]:
        """Child span recording start at __enter__, end at __exit__."""
        attrs = attributes_with_type(
            span_type,
            data_id=data_id if data_id is not None else self._task_id,
            extra={"batch_id": self._current_batch_id, **(attributes or {})},
        )
        with get_tracer().start_as_current_span(name, attributes=attrs) as span:
            yield span

    def _log_event(
        self,
        name: str,
        *,
        span_type: SpanType = SpanType.MARKER,
        data_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record a moment-in-time checkpoint as a zero-duration span."""
        attrs = attributes_with_type(
            span_type,
            data_id=data_id if data_id is not None else self._task_id,
            extra={"batch_id": self._current_batch_id, **(attributes or {})},
        )
        with get_tracer().start_as_current_span(name, attributes=attrs):
            pass

    # ------------------------------------------------------------------ #
    # Asset / lineage rows — keep their own JSONL files                  #
    # ------------------------------------------------------------------ #
    def _lineage_dir(self) -> Path:
        """Per-task ``logs/`` directory; requires an active ``_task_span``."""
        if self._task_out_dir is None:
            raise ExecutionError(
                "Lineage directory accessed before _task_span entered; "
                "wrap executor work in `with self._task_span(...)`."
            )
        return self._task_out_dir / "logs"

    def _append_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        target_dir = self._lineage_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, default=str)
        path = target_dir / filename
        with self._event_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _record_asset(
        self,
        data_id: str,
        asset_guid: str,
        version: int = 1,
        user_id: str = "",
        created_at: str | None = None,
    ) -> None:
        row = {
            "data_id": data_id,
            "asset_guid": asset_guid,
            "version": version,
            "user_id": user_id,
            "created_at": created_at or now_iso(),
        }
        self._append_jsonl("assets.jsonl", row)

    def _record_lineage(
        self,
        data_id: str,
        source_data_ids: Sequence[str],
        created_at: str | None = None,
    ) -> None:
        ts = created_at or now_iso()
        for source_data_id in source_data_ids:
            self._append_jsonl(
                "lineage.jsonl",
                {
                    "data_id": data_id,
                    "source_data_id": source_data_id,
                    "created_at": ts,
                },
            )

    def _record_output(
        self,
        data_id: str,
        data: Any,
        source_data_ids: list[str],
    ) -> None:
        """Emit asset + lineage rows; ``data`` is only serialized to size the
        ``"dump to storage"`` span (runtime does not upload payloads)."""
        with self._span(
            READY_SPAN_NAME, span_type=SpanType.NETWORK, data_id=data_id
        ) as dump_span:
            try:
                payload = json.dumps(data, ensure_ascii=False, default=str)
            except (TypeError, ValueError) as exc:
                raise ExecutionError(
                    f"Failed to serialize data {data_id}: {exc}"
                ) from exc

            payload_bytes = len(payload.encode("utf-8"))
            dump_span.set_attribute("payload_bytes", payload_bytes)

            asset_guid = (
                source_data_ids[0] if len(source_data_ids) == 1 else str(uuid.uuid4())
            )
            self._record_asset(
                data_id=data_id,
                asset_guid=asset_guid,
                version=1,
                user_id=self._task_owner_id,
            )
            if source_data_ids:
                self._record_lineage(data_id=data_id, source_data_ids=source_data_ids)
        logger.info(
            "Wrote lineage for %s (size: %d bytes, sources: %d)",
            data_id,
            payload_bytes,
            len(source_data_ids),
        )

    @staticmethod
    def _spec_upstream_results(spec: TaskSpecStrictBase) -> dict[str, Any]:
        """Validated ``spec._upstreamResults`` (server-injected stage context)."""
        context = spec.upstreamResults or {}
        if not isinstance(context, dict):
            raise ExecutionError("spec._upstreamResults must be a mapping.")
        return context

    def _extract_source_data_ids(self, spec: TaskSpecStrictBase) -> list[str]:
        """Extract upstream task/data IDs from ``_upstreamResults`` for lineage."""
        seen: set[str] = set()
        ids: list[str] = []
        for upstream in self._spec_upstream_results(spec).values():
            if not isinstance(upstream, dict):
                continue
            candidate = upstream.get("task_id") or upstream.get("data_id")
            if candidate is None:
                continue
            sid = str(candidate)
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
        return ids

    def _dump_to_governance(
        self,
        task_id: str,
        result: BaseExecutorResult | dict[str, Any],
        dependencies_by_task: dict[str, list[str]],
    ) -> None:
        """Write parent + merged-child results and emit asset/lineage rows."""
        parent_deps = dependencies_by_task.get(task_id, [])
        payload = (
            result.model_dump() if isinstance(result, BaseExecutorResult) else result
        )
        children_payload = payload.get("children", {}) or {}

        collection_jobs: list[dict[str, Any]] = [
            {
                "task_id": task_id,
                "result": payload,
                "deps": parent_deps,
                "is_parent": True,
            }
        ]
        for child_id, child_result in children_payload.items():
            child_deps = dependencies_by_task.get(child_id, [])
            collection_jobs.append(
                {
                    "task_id": child_id,
                    "result": child_result,
                    "deps": child_deps,
                    "is_parent": False,
                }
            )

        if len(collection_jobs) == 1:
            job = collection_jobs[0]
            self._record_output(
                data_id=job["task_id"],
                data=job["result"],
                source_data_ids=job["deps"],
            )
        else:
            logger.info(
                "Recording lineage for %d merged tasks in parallel",
                len(collection_jobs),
            )
            future_map = {
                self._submit_in_context(
                    self._record_output,
                    data_id=job["task_id"],
                    data=job["result"],
                    source_data_ids=job["deps"],
                ): job
                for job in collection_jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    if job["is_parent"]:
                        raise
                    raise ExecutionError(
                        f"Failed to write merged child task {job['task_id']}: {exc}"
                    ) from exc
