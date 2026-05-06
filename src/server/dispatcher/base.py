import copy
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.schemas.event import TaskEvent
from shared.tasks import (
    MergedChildTaskStrict,
    TaskEnvelope,
    TaskEnvelopeStrict,
    TaskEnvelopeTemplate,
)
from shared.tasks.placeholders import PLACEHOLDER_PATTERN
from shared.tasks.specs import (
    ConditionSpec,
    SSHSpecStrict,
    SSHSpecTemplate,
)
from shared.tasks.worker_message import WorkerStatus, WorkerTaskMessage

from ..registries.worker import Worker, WorkerRegistry
from ..schemas.result import result_file_path, write_result
from ..services.metrics import MetricsRecorder
from ..task.metadata import extract_model_dataset_names
from ..task.models import TaskRecord, TaskStatus
from ..task.runtime import TaskRuntime
from ..utils.time import now_iso
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker


class StageReferenceNotReady(Exception):
    """Raised when a task references a stage whose artifacts are not yet available."""


class Dispatcher:
    """Handles FCFS task dispatching via Redis pub/sub."""

    def __init__(
        self,
        runtime: TaskRuntime,
        worker_registry: WorkerRegistry,
        results_dir: Path,
        logger: logging.Logger,
        worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
        enable_context_reuse: bool = True,
        enable_task_merge: bool = True,
        task_merge_max_batch_size: int = 4,
        reuse_cache_ttl_sec: int = 3600,
        lambda_config: dict[str, float] | None = None,
        selection_jitter_epsilon: float = 1e-3,
        enable_stage_weight_stickiness: bool = False,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> None:
        self._runtime = runtime
        self._worker_registry = worker_registry
        self._logger = logger
        self._results_dir = Path(results_dir)
        self._worker_selection_strategy = worker_selection_strategy
        self._context_reuse_enabled = enable_context_reuse
        self._task_merge_enabled = enable_task_merge
        self._task_merge_max_batch_size = max(1, task_merge_max_batch_size)
        self._cache_ttl_sec = max(0, int(reuse_cache_ttl_sec))
        self._lambda_config = lambda_config or {}
        self._selection_jitter = max(0.0, float(selection_jitter_epsilon))
        self._stage_weight_stickiness_enabled = bool(enable_stage_weight_stickiness)
        self._metrics = metrics_recorder
        self._weight_reference_hints: tuple[str, ...] = (
            "checkpoint",
            "weight",
            "weights",
            "model",
            "adapter",
            "lora",
            "load",
            "artifact",
        )

    def dispatch_once(self, task_id: str) -> bool:
        """Dispatch a single task if possible; requeue when no worker."""
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        task = record.task

        model_names, dataset_names = extract_model_dataset_names(task)
        task_category = (
            (record.category or "other").lower()
            if hasattr(record, "category")
            else "other"
        )
        task_age = None
        if getattr(record, "last_queue_ts", None) is not None:
            task_age = max(0.0, time.time() - record.last_queue_ts)

        # 1. Get idle worker pool
        pool = self._worker_registry.idle_satisfying_pool(task)

        # 2. Filter by selected_worker hint if present
        if record.selected_worker:
            pool = [c for c in pool if c.id in record.selected_worker]

        # 3. If no workers -> requeue
        if not pool:
            reason = (
                "candidate_workers_busy" if record.selected_worker else "no_idle_worker"
            )
            if record.selected_worker:
                self._logger.debug(
                    "No candidate worker available for %s; allowed=%s",
                    task_id,
                    sorted(record.selected_worker),
                )
            else:
                self._logger.debug(
                    "No idle worker available for %s; requeueing", task_id
                )
            self._requeue_task(task_id, reason=reason, count_retry=False)
            return False

        # 4. Filter out last_failed_worker
        last_failed_worker = (getattr(record, "last_failed_worker", None) or "").strip()
        if last_failed_worker:
            filtered_pool = [
                candidate for candidate in pool if candidate.id != last_failed_worker
            ]
            if not filtered_pool:
                # Only available worker is the one that previously failed.
                # Requeue and count as a retry so max_attempts is respected;
                # the task will be failed once attempts are exhausted.
                self._logger.info(
                    "Only worker %s available for %s (previously failed); "
                    "requeueing with retry count",
                    last_failed_worker,
                    task_id,
                )
                record.last_failed_worker = None
                self._requeue_task(
                    task_id, reason="only_failed_worker_available", count_retry=True
                )
                return False
            if len(filtered_pool) != len(pool):
                self._logger.debug(
                    "Excluding last failed worker %s from candidate pool for %s "
                    "(size %d -> %d)",
                    last_failed_worker,
                    task_id,
                    len(pool),
                    len(filtered_pool),
                )
            pool = filtered_pool

        # 5. Worker selection (best-fit scoring by default)
        selection_info: dict[str, Any] = {}
        worker: Worker | None = None

        sticky_worker_id: str | None = None
        if (not record.selected_worker) and self._stage_weight_stickiness_enabled:
            sticky_worker_id = self._preferred_stage_worker(record, task)
            selection_info = {
                "strategy": "stage_weight_affinity",
                "sticky_worker": sticky_worker_id,
                "candidate_pool": len(pool),
                "chosen_metrics": {"score": 1.0},
            }

        if sticky_worker_id:
            sticky_candidate = next(
                (item for item in pool if item.id == sticky_worker_id), None
            )
            if sticky_candidate:
                worker = sticky_candidate
                self._logger.debug(
                    "Using sticky worker %s for %s",
                    sticky_worker_id,
                    task_id,
                )
            else:
                if self._should_wait_for_sticky_worker(record, sticky_worker_id):
                    self._logger.debug(
                        "Preferred worker %s busy for %s; waiting for availability",
                        sticky_worker_id,
                        task_id,
                    )
                    self._requeue_task(
                        task_id, reason="sticky_worker_busy", count_retry=False
                    )
                    return False
                else:
                    self._logger.debug(
                        "Preferred worker %s unavailable for %s; falling "
                        "back to normal selection",
                        sticky_worker_id,
                        task_id,
                    )

        preferred_pool: list[Worker] = []
        if worker is None and self._context_reuse_enabled:
            preferred_pool = self._cached_worker_candidates(
                pool, model_names, dataset_names
            )
        if worker is None:
            candidate_pool = preferred_pool or pool
            if preferred_pool:
                self._logger.debug(
                    "Preferring cached worker candidates for %s "
                    "(models=%s datasets=%s)",
                    task_id,
                    model_names,
                    dataset_names,
                )

            worker, selection_info = select_worker(
                candidate_pool,
                self._worker_selection_strategy,
                logger=self._logger,
                task_category=task_category,
                lambda_overrides=self._lambda_config,
                task_id=task_id,
                jitter_epsilon=self._selection_jitter,
                task_age=task_age,
            )
            if not worker and preferred_pool:
                worker, selection_info = select_worker(
                    pool,
                    self._worker_selection_strategy,
                    logger=self._logger,
                    task_category=task_category,
                    lambda_overrides=self._lambda_config,
                    task_id=task_id,
                    jitter_epsilon=self._selection_jitter,
                    task_age=task_age,
                )
        if not worker:
            self._logger.debug(
                "No suitable worker selected for %s; requeueing", task_id
            )
            self._requeue_task(task_id, reason="no_selection")
            return False

        # 5.5. Plan task merge: coalesce sibling merge candidates onto this worker
        merged_children: list[str] = []
        if self._task_merge_enabled and self._task_merge_max_batch_size > 1:
            merged_children = self._runtime.plan_merge(
                task_id, self._task_merge_max_batch_size, worker.id
            )
            if merged_children:
                self._logger.debug(
                    "Coalesced task %s with siblings %s",
                    task_id,
                    ", ".join(merged_children),
                )

        # 6. Resolve stage references
        try:
            rendered_task = self._resolve_stage_references(task_id, task, record)
        except StageReferenceNotReady as exc:
            self._logger.debug("Task %s waiting on stage artifacts: %s", task_id, exc)
            self._requeue_task(
                task_id, reason="stage_reference_pending", count_retry=False
            )
            return False
        except ValidationError as exc:
            self._runtime.release_merge(task_id)
            self._fail_task(
                task_id,
                "task_schema_validation_failed",
                payload={"error": str(exc)},
            )
            return True
        except Exception as exc:
            self._logger.error(
                "Failed to resolve stage references for %s: %s", task_id, exc
            )
            self._runtime.release_merge(task_id)
            self._fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        # 6.5. Conditional execution: skip dispatch if condition not met
        if self._evaluate_condition_skip(task_id, rendered_task, record):
            return True

        # 6.6. Resolve merged-child rendered payloads
        rendered_children: list[MergedChildTaskStrict] | None = None
        if record.merged_children:
            rendered_children = []
            for child_id in record.merged_children:
                if not child_id:
                    self._runtime.release_merge(task_id)
                    self._fail_task(
                        task_id,
                        "merged_child_missing_task_id",
                        payload={"error": "Merged child entry missing task_id"},
                    )
                    return True
                child_record = self._runtime.get_record(child_id)
                if not child_record:
                    self._runtime.release_merge(task_id)
                    self._fail_task(
                        task_id,
                        "merged_child_missing_record",
                        payload={"error": f"Merged child record missing: {child_id}"},
                    )
                    return True
                try:
                    resolved_child_task = self._resolve_stage_references(
                        child_id, child_record.task, child_record
                    )
                except StageReferenceNotReady as exc:
                    self._logger.debug(
                        "Merged child %s waiting on stage artifacts: %s",
                        child_id,
                        exc,
                    )
                    self._runtime.release_merge(task_id)
                    self._requeue_task(
                        task_id, reason="stage_reference_pending", count_retry=False
                    )
                    return False
                except Exception as exc:
                    self._logger.error(
                        "Failed to resolve stage references for merged child %s: %s",
                        child_id,
                        exc,
                    )
                    self._runtime.release_merge(task_id)
                    self._fail_task(
                        task_id,
                        f"Failed to resolve merged child {child_id}: {exc}",
                        payload={"error": str(exc)},
                    )
                    return True
                try:
                    rendered_children.append(
                        MergedChildTaskStrict(
                            task_id=child_id,
                            owner_id=child_record.owner_id,
                            workflow_id=child_record.workflow_id,
                            spec=resolved_child_task.spec,
                            metadata=resolved_child_task.metadata,
                        )
                    )
                except ValidationError as exc:
                    self._runtime.release_merge(task_id)
                    self._fail_task(
                        task_id,
                        "merged_child_schema_validation_failed",
                        payload={"error": str(exc), "child_task_id": child_id},
                    )
                    return True

        # 7. Build WorkerTaskMessage
        message = WorkerTaskMessage(
            task_id=task_id,
            workflow_id=record.workflow_id,
            owner_id=record.owner_id,
            task=rendered_task,
            task_type=record.task_type,
            assigned_worker=worker.id,
            dispatched_at=now_iso(),
            parent_task_id=None,
            shard_index=record.shard_index,
            shard_total=record.shard_total,
            merged_children=rendered_children,
            upstream_task_ids=self._resolve_upstream_task_ids(
                record, rendered_task.spec
            ),
        )

        # 8. Publish task
        try:
            receivers = self._worker_registry.publish_task(worker, message)
        except Exception as exc:
            self._logger.warning("Failed to publish task %s: %s", task_id, exc)
            self._requeue_task(task_id, reason="publish_failed", front=True)
            return False

        if receivers <= 0:
            self._logger.info(
                "No subscribers on tasks channel; delaying task %s", task_id
            )
            self._requeue_task(task_id, reason="no_subscribers", front=True)
            return False

        # 9. Mark dispatched
        self._runtime.mark_dispatched(task_id, worker)
        if merged_children:
            try:
                self._logger.info(
                    "[TaskMerge] parent=%s merged_children=%d -> %s",
                    task_id,
                    len(merged_children),
                    ", ".join(merged_children),
                )
            except Exception:
                pass
        try:
            self._worker_registry.update_worker_status(worker.id, WorkerStatus.BUSY)
        except Exception as exc:
            self._logger.debug("Failed to update worker %s status: %s", worker.id, exc)

        try:
            chosen_score = selection_info.get("chosen_metrics", {}).get("score")
            score_display = (
                f"{chosen_score:.4f}"
                if isinstance(chosen_score, (int, float))
                else "n/a"
            )
            self._logger.info(
                "Dispatch %s -> %s (strategy=%s score=%s age=%.2fs)",
                task_id,
                worker.id,
                selection_info.get("strategy"),
                score_display,
                task_age if task_age is not None else -1.0,
            )
        except Exception:
            pass
        return True

    def dispatch_loop(self, stop_event, poll_interval: float = 1.0) -> None:
        """Continuously dispatch ready tasks until stop_event is set."""
        while not stop_event.is_set():
            task_id = self._runtime.next_ready(stop_event, timeout=poll_interval)
            if not task_id:
                continue
            try:
                success = self.dispatch_once(task_id)
                if not success:
                    time.sleep(0.5)
            except Exception as exc:
                self._logger.exception("Dispatch loop error for %s: %s", task_id, exc)
                self._requeue_task(task_id, reason="dispatch_exception", front=True)
                time.sleep(1.0)

    def _requeue_task(
        self,
        task_id: str,
        *,
        reason: str,
        front: bool = False,
        release_merge: bool = True,
        mark_pending: bool = True,
        count_retry: bool = True,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if release_merge:
            self._runtime.release_merge(task_id)

        if mark_pending:
            self._runtime.mark_pending(task_id, increment_retry=count_retry)

        record = self._runtime.get_record(task_id)
        if count_retry and record:
            attempts = record.attempts
            max_attempts = record.max_attempts
            if (
                max_attempts is not None
                and max_attempts >= 0
                and attempts >= max_attempts
            ):
                self._fail_task(
                    task_id,
                    "max_attempts_exceeded",
                    payload={
                        "reason": reason,
                        "attempts": attempts,
                        "max_attempts": max_attempts,
                    },
                    worker_id=record.last_failed_worker or record.assigned_worker,
                )
                return

        added = self._runtime.requeue(task_id, front=front)
        if not added:
            self._logger.warning(
                "Task %s could not be re-added to ready queue after requeue "
                "(reason=%s); task may remain stuck in PENDING",
                task_id,
                reason,
            )
        if count_retry:
            payload = {"reason": reason}
            if extra_payload:
                payload.update(extra_payload)
            self._emit_task_event("TASK_REQUEUED", task_id, payload=payload)

    def _fail_task(
        self,
        task_id: str,
        error_message: str,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        failure_payload = payload.copy() if isinstance(payload, dict) else {}
        if error_message and "error" not in failure_payload:
            failure_payload["error"] = error_message
        impacted, merged_children, _ = self._runtime.mark_failed(
            task_id,
            worker_id,
            failure_payload,
            now_iso(),
            error=error_message,
        )
        self._emit_task_event(
            "TASK_FAILED",
            task_id,
            worker_id=worker_id,
            payload=failure_payload,
            error=error_message,
        )
        for dep_id, reason in impacted:
            dependent_payload = {"dependency_failure": task_id, "error": reason}
            self._emit_task_event(
                "TASK_FAILED",
                dep_id,
                payload=dependent_payload,
                error=reason,
            )
        for child_id in merged_children:
            child_payload = failure_payload.copy()
            child_payload["parent_task_id"] = task_id
            child_payload["dependency_failure"] = task_id
            child_payload["is_child_task"] = True
            self._emit_task_event(
                "TASK_FAILED",
                child_id,
                payload=child_payload,
                error=error_message or "parent_failed",
                is_child=True,
            )

    def _emit_task_event(
        self,
        event_type: str,
        task_id: str,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
        is_child: bool = False,
    ) -> None:
        if not self._metrics:
            return
        event = TaskEvent(
            type=event_type,
            task_id=task_id,
            worker_id=worker_id,
            payload=payload or {},
            error=error,
            ts=now_iso(),
        )
        self._metrics.record_task_event(event, is_child=is_child)
        if event_type == "TASK_FAILED":
            self._metrics.finalize_task_failure(task_id)

    def _cached_worker_candidates(
        self,
        pool: list[Worker],
        model_names: list[str],
        dataset_names: list[str],
    ) -> list[Worker]:
        if not pool or (not model_names and not dataset_names):
            return []

        def _score(worker: Worker) -> int:
            score = 0
            cached_models = {
                m.lower() for m in (worker.cached_models or []) if isinstance(m, str)
            }
            cached_datasets = {
                d.lower() for d in (worker.cached_datasets or []) if isinstance(d, str)
            }
            if model_names:
                score += sum(1 for name in model_names if name.lower() in cached_models)
            if dataset_names:
                score += sum(
                    1 for name in dataset_names if name.lower() in cached_datasets
                )
            return score

        scored: list[tuple[Worker, int]] = []
        ttl = self._cache_ttl_sec
        now_ts = time.time()
        for worker in pool:
            if ttl:
                ts_raw = worker.cache_updated_ts
                if not ts_raw:
                    continue
                try:
                    parsed = datetime.datetime.fromisoformat(
                        ts_raw.rstrip("Z").replace("Z", "+00:00")
                    )
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=datetime.UTC)
                except Exception:
                    continue
                age = now_ts - parsed.timestamp()
                if age > ttl:
                    continue
            value = _score(worker)
            if value > 0:
                scored.append((worker, value))

        if not scored:
            return []

        max_score = max(score for _, score in scored)
        return [worker for worker, score in scored if score == max_score]

    def _preferred_stage_worker(
        self, record: TaskRecord, task: TaskEnvelope
    ) -> str | None:
        if not record or not record.graph_node_name:
            return None
        task_type = (record.task_type or "").strip().lower()
        if task_type.startswith("lora"):
            return None
        try:
            info = self._runtime.describe_task(record.task_id)
            depends_on = info.depends_on if info else None
        except Exception:
            depends_on = None
        if not depends_on:
            return None
        for dep_id in depends_on:
            dep_record = self._runtime.get_record(dep_id)
            if not dep_record or not dep_record.assigned_worker:
                continue
            stage_refs = [dep_record.graph_node_name, dep_id]
            for ref in stage_refs:
                if not ref:
                    continue
                if self._spec_has_weight_reference(task, ref):
                    return dep_record.assigned_worker
        return None

    def _spec_has_weight_reference(
        self, task: TaskEnvelope, stage_identifier: str
    ) -> bool:
        token = f"${{{stage_identifier}.result"
        return self._search_weight_reference(task, token, tuple())

    def _search_weight_reference(
        self, value: Any, token: str, path: tuple[str, ...]
    ) -> bool:
        if isinstance(value, BaseModel):
            for key, sub in value:
                next_path = path + (str(key).lower(),)
                if self._search_weight_reference(sub, token, next_path):
                    return True
        elif isinstance(value, dict):
            for key, sub in value.items():
                next_path = path + (str(key).lower(),)
                if self._search_weight_reference(sub, token, next_path):
                    return True
        elif isinstance(value, list):
            for item in value:
                if self._search_weight_reference(item, token, path):
                    return True
        elif isinstance(value, str):
            if token in value:
                lowered = value.lower()
                if any(hint in path for hint in self._weight_reference_hints) or any(
                    hint in lowered for hint in self._weight_reference_hints
                ):
                    return True
        return False

    def _should_wait_for_sticky_worker(self, record, worker_id: str) -> bool:
        if not record or not worker_id:
            return False
        task_type = (record.task_type or "").strip().lower()
        if not task_type:
            return False
        if task_type.startswith("lora"):
            return False
        category = (record.category or "").strip().lower()
        if category != "training":
            return False
        try:
            worker = self._worker_registry.get_worker(worker_id)
        except Exception:
            worker = None
        if not worker:
            return False
        try:
            if self._worker_registry.is_worker_stale(worker_id):
                return False
        except Exception:
            return False
        if worker.status is WorkerStatus.IDLE:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Stage reference handling
    # ------------------------------------------------------------------ #

    def _resolve_stage_references(
        self, task_id: str, task: TaskEnvelopeTemplate, record: TaskRecord
    ) -> TaskEnvelopeStrict:
        context = self._build_stage_context(record)
        resolved_task = task
        if context and task.has_placeholder():
            resolved_task = self._resolve_placeholders(task, context)

        upstream_results = (
            self._collect_upstream_results(context, task_id) if context else {}
        )
        if upstream_results:
            existing = resolved_task.spec.upstreamResults or {}
            merged: dict[str, Any] = {}
            if isinstance(existing, dict):
                merged.update(copy.deepcopy(existing))
            merged.update(upstream_results)
            resolved_task = resolved_task.model_copy(
                update={
                    "spec": resolved_task.spec.model_copy(
                        update={"upstreamResults": merged}
                    )
                }
            )

        return TaskEnvelopeStrict.model_validate(resolved_task)

    def _resolve_placeholders(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, str):
            exact = PLACEHOLDER_PATTERN.fullmatch(value)
            if exact:
                return self._resolve_reference(exact.group(1), context)
            return PLACEHOLDER_PATTERN.sub(
                lambda m: str(self._resolve_reference(m.group(1), context)),
                value,
            )
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(item, context) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_placeholders(item, context) for item in value)
        if isinstance(value, BaseModel):
            updates: dict[str, Any] = {}
            for key, current in value:
                transformed = self._resolve_placeholders(current, context)
                if transformed is not current:
                    updates[key] = transformed
            return value.model_copy(update=updates) if updates else value
        return value

    def _contains_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(PLACEHOLDER_PATTERN.search(value))
        if isinstance(value, dict):
            return any(self._contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    def _build_stage_context(self, record: TaskRecord) -> dict[str, TaskRecord]:
        """Collect upstream dependency records keyed by stage identity."""
        if not self._needs_stage_context(record):
            return {}

        context: dict[str, TaskRecord] = {}
        for dep_id in self._dependency_task_ids(record.task_id):
            other = self._runtime.get_record(dep_id)
            if other is None:
                continue
            for key in self._stage_context_keys(other):
                context[key] = other
        return context

    def _needs_stage_context(self, record: TaskRecord) -> bool:
        if record.graph_node_name is not None or record.local_name is not None:
            return True
        if record.task.has_placeholder():
            return True
        spec = record.task.spec
        return isinstance(spec, (SSHSpecStrict, SSHSpecTemplate)) and bool(spec.inputs)

    def _dependency_task_ids(self, task_id: str) -> set[str]:
        pending = self._task_dependencies(task_id)
        visited: set[str] = set()
        # Walk the upstream dependency graph to collect all dependency tasks
        while pending:
            dep_id = pending.pop()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            pending.extend(
                upstream_id
                for upstream_id in self._task_dependencies(dep_id)
                if upstream_id not in visited
            )
        return visited

    def _task_dependencies(self, task_id: str) -> list[str]:
        info = self._runtime.describe_task(task_id)
        return [] if info is None else info.depends_on.copy()

    @staticmethod
    def _stage_context_keys(record: TaskRecord) -> tuple[str, ...]:
        keys: list[str] = []
        if record.local_name:
            keys.append(record.local_name)
        if record.graph_node_name and record.graph_node_name not in keys:
            keys.append(record.graph_node_name)
        return tuple(keys)

    def _resolve_reference(self, expr: str, context: dict[str, Any]) -> Any:
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty stage reference")
        if "." not in expr:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_name, path = expr.split(".", 1)
        stage_name = stage_name.strip()
        if not stage_name:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_record = context.get(stage_name)
        if not stage_record:
            raise ValueError(f"Unknown stage reference '{stage_name}'")
        if path == "task_id":
            return stage_record.task_id
        if stage_record.status != "DONE":
            raise StageReferenceNotReady(f"Stage '{stage_name}' has not completed")
        data = self._load_stage_result(stage_record.task_id)
        value = self._dig_path(data, path.split("."))
        if value is None:
            raise ValueError(f"Missing value for reference '{expr}'")
        # If the referenced value is an artifact ref ({path: "..."}), render
        # it as a full URL (when base_url is set) or an absolute filesystem
        # path using the producing stage's top-level _artifacts context.
        if rendered := self._render_artifact_ref(value, data):
            return rendered
        return value

    @staticmethod
    def _render_artifact_ref(value: Any, stage_result: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            return None
        if not isinstance(stage_result, dict):
            return None
        ctx = stage_result.get("_artifacts")
        if not isinstance(ctx, dict):
            return None
        base_url = ctx.get("base_url")
        base_dir = ctx.get("base_dir")
        if isinstance(base_url, str) and base_url and isinstance(base_dir, str):
            task_id = Path(base_dir).name
            return f"{base_url.rstrip('/')}/api/v1/results/{task_id}/files/{path_value}"
        if isinstance(base_dir, str) and base_dir:
            return (Path(base_dir) / "artifacts" / path_value).as_posix()
        return None

    def _collect_upstream_results(
        self, context: dict[str, TaskRecord], current_task_id: str
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, record in context.items():
            if not record or record.task_id == current_task_id:
                continue
            if record.status != TaskStatus.DONE:
                continue
            try:
                data = self._load_stage_result(record.task_id)
            except StageReferenceNotReady as exc:
                raise exc
            except Exception as exc:
                self._logger.debug(
                    "Failed to load upstream result for %s (%s): %s",
                    name,
                    record.task_id,
                    exc,
                )
                continue
            results[name] = {"result": data}
        return results

    def _resolve_upstream_task_ids(
        self, record: TaskRecord, spec: Any
    ) -> dict[str, str] | None:
        if not isinstance(spec, SSHSpecStrict) or not spec.inputs:
            return None

        context = self._build_stage_context(record)
        resolved: dict[str, str] = {}
        for entry in spec.inputs:
            stage_name = entry.stage.strip()
            if not stage_name:
                raise ValueError("SSH input stage names must be non-empty")
            upstream = context.get(stage_name)
            if upstream is None:
                raise ValueError(
                    f"Unknown SSH input stage '{stage_name}' for task {record.task_id}"
                )
            if upstream.task_id == record.task_id:
                raise ValueError(
                    f"SSH input stage '{stage_name}' cannot reference the current task"
                )
            if upstream.status != TaskStatus.DONE:
                raise StageReferenceNotReady(
                    f"Stage '{stage_name}' has not completed for SSH input mount"
                )
            resolved[stage_name] = upstream.task_id
        return resolved or None

    def _load_stage_result(self, stage_task_id: str) -> dict[str, Any]:
        path = result_file_path(self._results_dir, stage_task_id)
        if not path.exists():
            raise StageReferenceNotReady(
                f"Result for task {stage_task_id} not found at {path}"
            )
        content = json.loads(path.read_text(encoding="utf-8"))
        return content

    def _dig_path(self, data: Any, parts: list[str]) -> Any:
        current = data
        for part in parts:
            part = part.strip()
            if part == "":
                continue
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
                continue
            if isinstance(current, list):
                try:
                    idx = int(part)
                except ValueError as exc:
                    raise ValueError(
                        f"List index must be integer in reference path, got '{part}'"
                    ) from exc
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            return None
        return current

    def _evaluate_condition_skip(
        self,
        task_id: str,
        rendered_task: TaskEnvelopeStrict,
        record: TaskRecord,
    ) -> bool:
        """Evaluate a task's condition and skip it if the condition is not met.

        Returns ``True`` if the task was skipped (caller should return early),
        ``False`` if the task should proceed with normal dispatch.
        """
        condition: ConditionSpec | None = rendered_task.spec.condition
        if condition is None:
            return False

        try:
            stage_context = self._build_stage_context(record)
            upstream_record = stage_context.get(condition.node)
            if upstream_record is None:
                raise ValueError(
                    f"Condition references unknown node '{condition.node}'; "
                    f"known nodes: {list(stage_context.keys())}"
                )
            if upstream_record.status != TaskStatus.DONE:
                raise StageReferenceNotReady(
                    f"Condition upstream node '{condition.node}' not yet DONE "
                    f"(status={upstream_record.status})"
                )
            upstream_result = self._load_stage_result(upstream_record.task_id)
            actual_value = self._dig_path(upstream_result, condition.field.split("."))
            if str(actual_value) == condition.equals:
                return False  # Condition met — proceed with dispatch

            self._logger.info(
                "Task %s skipped: condition %s.%s == %r not met (actual=%r)",
                task_id,
                condition.node,
                condition.field,
                condition.equals,
                actual_value,
            )
            skip_result: dict[str, Any] = {
                "task_id": task_id,
                "result": {},
                "metadata": {
                    "skipped": True,
                    "reason": "condition_not_met",
                    "condition_node": condition.node,
                    "condition_field": condition.field,
                    "condition_expected": condition.equals,
                    "condition_actual": str(actual_value),
                },
            }
            write_result(self._results_dir, task_id, skip_result)
            self._runtime.release_merge(task_id)
            ts = now_iso()
            self._runtime.mark_succeeded(
                task_id,
                worker_id=None,
                payload={"finished_at": ts, "started_at": ts},
                ts=ts,
            )
            return True
        except StageReferenceNotReady as exc:
            self._logger.debug(
                "Task %s condition: upstream %s not ready: %s",
                task_id,
                condition.node,
                exc,
            )
            self._requeue_task(
                task_id, reason="condition_upstream_pending", count_retry=False
            )
            return True
        except Exception as exc:
            self._logger.error(
                "Failed to evaluate condition for task %s: %s", task_id, exc
            )
            self._runtime.release_merge(task_id)
            self._fail_task(
                task_id,
                f"condition_evaluation_failed: {exc}",
                payload={"error": str(exc)},
            )
            return True
