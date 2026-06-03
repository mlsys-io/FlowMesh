import copy
import heapq
import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any

from shared.schemas.command import InterruptMessage
from shared.tasks import TaskEnvelopeTemplate
from shared.utils import new_workflow_id

from ..hooks import SUPPLIER_RESOLVERS
from ..registries.worker import Worker, WorkerRegistry
from ..registries.workflow import PersistedTask, WorkflowRegistry, WorkflowSched
from ..utils.time import parse_iso_ts
from .models import (
    TERMINAL_TASK_STATUSES,
    TaskInfo,
    TaskParsingResult,
    TaskRecord,
    TaskStatus,
    TaskUsage,
    categorize_task_type,
)
from .parser import parse_workflow


def _sanitize_merge_spec(spec: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(spec)
    if isinstance(clone.get("inference"), dict):
        inference_cfg = clone["inference"]
        inference_cfg.pop("system_prompt", None)
    clone.pop("data", None)
    return clone


def _compute_merge_key(task: TaskEnvelopeTemplate) -> str | None:
    task_type = str(task.spec.taskType or "").strip().lower()
    if task_type not in {"inference", "rag", "diffusion"}:
        return None
    try:
        spec = task.spec.model_dump(mode="python", exclude_none=True)
        sanitized = _sanitize_merge_spec(spec)
        return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None


class TaskRuntime:
    """In-memory task registry with FIFO-ready queue and dependency tracking."""

    def __init__(
        self,
        workflow_registry: WorkflowRegistry,
        worker_registry: WorkerRegistry,
        logger: logging.Logger,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._worker_registry = worker_registry
        self._logger = logger
        self._tasks: dict[str, TaskRecord] = {}
        self._original_deps: dict[str, set[str]] = {}
        self._pending_deps: dict[str, set[str]] = {}
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._ready_by_workflow: dict[str, list[tuple[int, str]]] = {}
        self._ready_queue: deque[tuple[str, bool]] = (
            deque()
        )  # task_id | workflow_id, is_workflow
        self._ready_index: set[str] = set()
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._merge_key_by_task: dict[str, tuple[str | None, str | None]] = {}
        self._merge_buckets: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        self._merge_children_map: dict[str, list[str]] = defaultdict(list)
        self._merge_parent_map: dict[str, str] = {}
        self._workflow_epoch_tasks: dict[str, deque[set[str]]] = {}
        self._workflow_epoch_frontier: dict[str, int] = {}
        self._workflow_in_epoch_order: dict[str, bool] = {}
        self._task_epoch_index: dict[str, int] = {}
        self._rehydrated_dispatched: dict[str, float] = {}

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

    # ------------------------------------------------------------------ #
    # Durable state persistence (for restart rehydration)
    # ------------------------------------------------------------------ #

    def _persisted_task_locked(self, task_id: str) -> PersistedTask | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return PersistedTask(
            record=record,
            depends_on=self._original_deps.get(task_id, set()),
            epoch_index=self._task_epoch_index.get(task_id),
        )

    def _persist_locked(self, *task_ids: str) -> None:
        items = [
            persisted
            for task_id in dict.fromkeys(task_ids)
            if (persisted := self._persisted_task_locked(task_id)) is not None
        ]
        if items:
            self._workflow_registry.save_task_states(items)

    def _persist_sched_locked(self, workflow_id: str) -> None:
        self._workflow_registry.save_workflow_sched(
            workflow_id,
            self._workflow_in_epoch_order.get(workflow_id, False),
            self._workflow_epoch_frontier.get(workflow_id, 0),
        )

    # ------------------------------------------------------------------ #
    # Registration & submission
    # ------------------------------------------------------------------ #

    def validate(self, payload: str, format: str = "native") -> list[TaskParsingResult]:
        parsed_workflow = parse_workflow(payload, format)
        specs = parsed_workflow.tasks
        results: list[TaskParsingResult] = []
        for entry in specs:
            task_id = entry.task_id
            depends_on = entry.depends_on.copy()
            results.append(
                TaskParsingResult(
                    task_id=task_id,
                    graph_node_name=entry.graph_node_name,
                    depends_on=depends_on,
                )
            )
        return results

    async def register(
        self, owner_id: str, org_id: str, payload: str, format: str = "native"
    ) -> tuple[str, list[TaskParsingResult]]:
        parsed_workflow = parse_workflow(payload, format)
        specs = parsed_workflow.tasks
        yaml_text = payload
        results: list[TaskParsingResult] = []
        workflow_id = new_workflow_id()
        task_records: list[TaskRecord] = []
        candidate_ready: list[str] = []
        graph_task_ids: dict[str, str] = {}

        with self._cv:
            if (
                parsed_workflow.schedule_in_epoch_order
                and parsed_workflow.epoch_groups is not None
            ):
                self._ready_by_workflow[workflow_id] = []
                self._workflow_in_epoch_order[workflow_id] = True
            for entry in specs:
                task_id = entry.task_id
                task = entry.task.model_copy(deep=True)
                depends_on = entry.depends_on.copy()
                original = set(depends_on)
                pending = {dep for dep in depends_on if dep not in self._completed}

                task_type = task.spec.taskType
                category = categorize_task_type(task_type)

                selected_worker_raw = entry.selected_worker
                selected_worker: list[str] | None
                if isinstance(selected_worker_raw, list):
                    normalized_workers = [
                        str(worker_id).strip()
                        for worker_id in selected_worker_raw
                        if str(worker_id).strip()
                    ]
                    selected_worker = list(dict.fromkeys(normalized_workers)) or None
                elif isinstance(selected_worker_raw, str):
                    selected_worker = (
                        [selected_worker_raw.strip()]
                        if selected_worker_raw.strip()
                        else None
                    )
                else:
                    selected_worker = None

                record = TaskRecord(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    owner_id=owner_id,
                    raw_yaml=yaml_text,
                    task=task,
                    local_name=entry.local_name,
                    graph_node_name=entry.graph_node_name,
                    load=entry.load,
                    position_in_epoch=entry.position_in_epoch,
                    selected_worker=selected_worker,
                    task_type=task_type,
                    category=category,
                )
                task_records.append(record)
                record.last_queue_ts = record.submitted_ts
                merge_key = _compute_merge_key(task)
                record.merge_key = merge_key
                selected_worker_hint = (
                    record.selected_worker[0]
                    if record.selected_worker and len(record.selected_worker) == 1
                    else None
                )
                self._merge_key_by_task[task_id] = (merge_key, selected_worker_hint)

                self._tasks[task_id] = record
                if record.graph_node_name:
                    graph_task_ids[record.graph_node_name] = task_id
                self._original_deps[task_id] = original
                self._pending_deps[task_id] = pending
                self._failed.discard(task_id)
                if record.status == TaskStatus.DONE:
                    self._completed.add(task_id)
                else:
                    self._completed.discard(task_id)

                for dep in original:
                    self._dependents[dep].add(task_id)

                if not pending and record.status == TaskStatus.PENDING:
                    candidate_ready.append(task_id)

                results.append(
                    TaskParsingResult(
                        task_id=task_id,
                        graph_node_name=entry.graph_node_name,
                        depends_on=depends_on,
                    )
                )
            epoch_groups = parsed_workflow.epoch_groups
            if epoch_groups:
                epoch_queue: deque[set[str]] = deque()
                has_epoch_tasks = False
                for epoch_idx, epoch_nodes in enumerate(epoch_groups):
                    epoch_task_ids: set[str] = set()
                    for node_name in epoch_nodes:
                        mapped_task_id = graph_task_ids.get(node_name)
                        if mapped_task_id is None:
                            continue
                        epoch_task_ids.add(mapped_task_id)
                        self._task_epoch_index[mapped_task_id] = epoch_idx
                        has_epoch_tasks = True
                    epoch_queue.append(epoch_task_ids)
                if has_epoch_tasks:
                    self._workflow_epoch_tasks[workflow_id] = epoch_queue
                    self._workflow_epoch_frontier[workflow_id] = 0

        await self._workflow_registry.register_workflow_async(workflow_id, task_records)

        with self._cv:
            persisted = [
                item
                for record in task_records
                if (item := self._persisted_task_locked(record.task_id)) is not None
            ]
            in_epoch_order = self._workflow_in_epoch_order.get(workflow_id, False)
            frontier = self._workflow_epoch_frontier.get(workflow_id, 0)
        await self._workflow_registry.save_task_states_async(persisted)
        await self._workflow_registry.save_workflow_sched_async(
            workflow_id, in_epoch_order, frontier
        )

        new_ready = False
        with self._cv:
            for task_id in candidate_ready:
                maybe_record = self._tasks.get(task_id)
                if not maybe_record or maybe_record.status != TaskStatus.PENDING:
                    continue
                if self._pending_deps.get(task_id):
                    continue
                if self._enqueue_ready_locked(task_id):
                    new_ready = True
            if new_ready:
                self._cv.notify_all()

        return workflow_id, results

    # ------------------------------------------------------------------ #
    # Rehydration (restart recovery)
    # ------------------------------------------------------------------ #

    def rehydrate(self) -> int:
        """Rebuild in-memory scheduler state from durable Redis records.

        Reconstructs every live workflow's DAG, ready queue, and epoch state
        from the persisted per-task snapshots. In-flight (DISPATCHED /
        CANCELLING) tasks are left assigned to their worker: completions that
        landed during the restart arrive via the replayed task-event stream,
        and genuinely departed workers are recovered by the watchdog. Returns
        the number of workflows restored.
        """
        workflow_ids = self._workflow_registry.get_workflow_ids()
        rehydrated_at = time.time()
        restored = 0
        for workflow_id in sorted(workflow_ids):
            wf_record = self._workflow_registry.get_workflow_record(workflow_id)
            if wf_record is None:
                continue
            tasks: list[PersistedTask] = []
            for task_id in wf_record.task_ids:
                state = self._workflow_registry.load_task_state(task_id)
                if state is not None:
                    tasks.append(state)
            if not tasks:
                continue
            sched = self._workflow_registry.load_workflow_sched(workflow_id)
            with self._cv:
                self._install_rehydrated_workflow_locked(
                    workflow_id, tasks, sched, rehydrated_at
                )
                self._cv.notify_all()
            restored += 1
        if restored:
            self._logger.info("Rehydrated %d workflow(s) from durable state", restored)
        return restored

    def _install_rehydrated_workflow_locked(
        self,
        workflow_id: str,
        tasks: list[PersistedTask],
        sched: "WorkflowSched | None",
        rehydrated_at: float,
    ) -> None:
        terminal = TERMINAL_TASK_STATUSES
        in_epoch_order = bool(sched.in_epoch_order) if sched else False
        frontier = int(sched.epoch_frontier) if sched else 0
        epoch_members: dict[int, set[str]] = defaultdict(set)

        for persisted in tasks:
            record = persisted.record
            task_id = record.task_id
            self._tasks[task_id] = record
            self._original_deps[task_id] = set(persisted.depends_on)
            selected_worker_hint = (
                record.selected_worker[0]
                if record.selected_worker and len(record.selected_worker) == 1
                else None
            )
            self._merge_key_by_task[task_id] = (record.merge_key, selected_worker_hint)
            if persisted.epoch_index is not None:
                self._task_epoch_index[task_id] = persisted.epoch_index
                epoch_members[persisted.epoch_index].add(task_id)
            if record.status == TaskStatus.DONE:
                self._completed.add(task_id)
            elif record.status == TaskStatus.FAILED:
                self._failed.add(task_id)
            elif record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                self._rehydrated_dispatched[task_id] = rehydrated_at

        for persisted in tasks:
            record = persisted.record
            task_id = record.task_id
            original = self._original_deps.get(task_id, set())
            for dep in original:
                self._dependents[dep].add(task_id)
            if record.status in terminal:
                continue
            # Only completed deps are subtracted, not failed ones: a failure
            # cascade-fails its dependents and persists them FAILED atomically,
            # so a non-terminal task here never has a FAILED dep to clear.
            self._pending_deps[task_id] = {
                dep for dep in original if dep not in self._completed
            }

        if in_epoch_order:
            self._workflow_in_epoch_order[workflow_id] = True
            self._ready_by_workflow.setdefault(workflow_id, [])
        if epoch_members:
            epoch_queue: deque[set[str]] = deque(
                epoch_members[idx] for idx in sorted(epoch_members) if idx >= frontier
            )
            if epoch_queue:
                self._workflow_epoch_tasks[workflow_id] = epoch_queue
                self._workflow_epoch_frontier[workflow_id] = frontier

        for persisted in tasks:
            record = persisted.record
            if record.status != TaskStatus.PENDING:
                continue
            if self._pending_deps.get(record.task_id):
                continue
            self._enqueue_ready_locked(record.task_id)

    # ------------------------------------------------------------------ #
    # Ready queue helpers
    # ------------------------------------------------------------------ #

    def _enqueue_ready_locked(self, task_id: str, *, front: bool = False) -> bool:
        """Add a task to the ready queue if it is pending and not already queued."""
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return False
        if task_id in self._ready_index:
            return False
        if not self._is_epoch_ready_locked(record):
            return False
        workflow_id = record.workflow_id
        if (
            workflow_id in self._workflow_in_epoch_order
            and task_id in self._task_epoch_index
        ):
            queue = self._ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            heapq.heappush(queue, (position_in_epoch, task_id))
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        if front:
            self._ready_queue.appendleft(ready_entry)
        else:
            self._ready_queue.append(ready_entry)
        self._ready_index.add(task_id)
        record.last_queue_ts = time.time()
        self._merge_bucket_add(task_id)
        return True

    def _pop_ready_locked(self) -> str | None:
        while self._ready_queue:
            task_or_workflow_id, is_workflow = self._ready_queue.popleft()
            if is_workflow:
                _, task_id = heapq.heappop(self._ready_by_workflow[task_or_workflow_id])
            else:
                task_id = task_or_workflow_id
            self._ready_index.discard(task_id)
            record = self._tasks.get(task_id)
            if not record or record.status != TaskStatus.PENDING:
                continue
            return task_id
        return None

    def _remove_from_ready_locked(self, task_id: str) -> None:
        if task_id not in self._ready_index:
            return
        record = self._tasks.get(task_id)
        if not record:
            return
        workflow_id = record.workflow_id
        if (
            workflow_id in self._workflow_in_epoch_order
            and task_id in self._task_epoch_index
        ):
            queue = self._ready_by_workflow[workflow_id]
            position_in_epoch = record.position_in_epoch
            if position_in_epoch is None:
                raise ValueError(
                    "Ordered workflow task is missing position_in_epoch "
                    f"(task_id={task_id})"
                )
            queue.remove((position_in_epoch, task_id))
            heapq.heapify(queue)
            ready_entry = (workflow_id, True)
        else:
            ready_entry = (task_id, False)
        self._ready_queue.remove(ready_entry)
        self._ready_index.discard(task_id)

    def _merge_bucket_add(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self._merge_buckets.setdefault((merge_key, selected_worker), [])
        if task_id not in bucket:
            bucket.append(task_id)

    def _merge_bucket_remove(self, task_id: str) -> None:
        key = self._merge_key_by_task.get(task_id)
        if not key:
            return
        merge_key, selected_worker = key
        if not merge_key:
            return
        bucket = self._merge_buckets.get((merge_key, selected_worker))
        if not bucket:
            return
        try:
            bucket.remove(task_id)
        except ValueError:
            pass
        if not bucket:
            self._merge_buckets.pop((merge_key, selected_worker), None)

    def _is_epoch_ready_locked(self, record: TaskRecord) -> bool:
        epoch_index = self._task_epoch_index.get(record.task_id)
        if epoch_index is None:
            return True
        frontier = self._workflow_epoch_frontier.get(record.workflow_id)
        if frontier is None:
            return True
        return epoch_index == frontier

    def _try_advance_epoch_frontier_locked(self, workflow_id: str) -> list[str]:
        epoch_tasks = self._workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return []
        frontier = self._workflow_epoch_frontier[workflow_id]

        ready: list[str] = []
        while True:
            self._workflow_epoch_frontier[workflow_id] = frontier
            current_tasks = epoch_tasks[0] if epoch_tasks else set()
            if current_tasks and not all(
                (task := self._tasks.get(task_id)) is not None
                and task.status == TaskStatus.DONE
                for task_id in current_tasks
            ):
                break

            if epoch_tasks:
                epoch_tasks.popleft()
            frontier += 1
            self._workflow_epoch_frontier[workflow_id] = frontier
            if not epoch_tasks:
                self._workflow_epoch_frontier.pop(workflow_id, None)
                self._workflow_epoch_tasks.pop(workflow_id, None)
                break

            for task_id in epoch_tasks[0]:
                record = self._tasks.get(task_id)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                if self._pending_deps.get(task_id):
                    continue
                if self._enqueue_ready_locked(task_id):
                    ready.append(task_id)

        return ready

    def _fail_later_epochs_locked(
        self,
        workflow_id: str,
        failed_epoch: int,
        reason: str,
    ) -> list[tuple[str, str]]:
        epoch_tasks = self._workflow_epoch_tasks.get(workflow_id)
        if not epoch_tasks:
            return []
        frontier = self._workflow_epoch_frontier[workflow_id]

        impacted: list[tuple[str, str]] = []
        for offset, epoch_task_ids in enumerate(epoch_tasks):
            epoch = frontier + offset
            if epoch <= failed_epoch:
                continue
            for task_id in epoch_task_ids:
                record = self._tasks.get(task_id)
                if not record or record.status != TaskStatus.PENDING:
                    continue
                record.status = TaskStatus.FAILED
                record.error = reason
                record.assigned_worker = None
                record.finished_ts = time.time()
                self._failed.add(task_id)
                self._completed.discard(task_id)
                self._pending_deps.pop(task_id, None)
                self._remove_from_ready_locked(task_id)
                self._merge_bucket_remove(task_id)
                self._merge_key_by_task.pop(task_id, None)
                self._merge_parent_map.pop(task_id, None)
                self._merge_children_map.pop(task_id, None)
                self._workflow_registry.mark_task_failed(workflow_id, task_id)
                impacted.append((task_id, reason))

        return impacted

    def next_ready(
        self, stop_event: threading.Event, timeout: float = 1.0
    ) -> str | None:
        """
        Block until a task is ready or stop_event is set.
        Returns a task_id or None when stopping.
        """
        with self._cv:
            while not stop_event.is_set():
                task_id = self._pop_ready_locked()
                if task_id:
                    return task_id
                self._cv.wait(timeout)
            return None

    def mark_pending(self, task_id: str, *, increment_retry: bool = False) -> None:
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status == TaskStatus.CANCELLED:
                return
            record.status = TaskStatus.PENDING
            record.assigned_worker = None
            record.topic = None
            record.dispatched_ts = None
            record.started_ts = None
            record.finished_ts = None
            record.error = None
            self._workflow_registry.mark_task_pending(record.workflow_id, task_id)
            if increment_retry:
                try:
                    if record.max_attempts is not None and record.max_attempts >= 0:
                        record.attempts = min(record.attempts + 1, record.max_attempts)
                    else:
                        record.attempts = record.attempts + 1
                except Exception:
                    record.attempts = (record.attempts or 0) + 1
            self._persist_locked(task_id)

    def requeue(self, task_id: str, *, front: bool = False) -> bool:
        """Reinsert a task into the ready queue."""
        with self._cv:
            added = self._enqueue_ready_locked(task_id, front=front)
            if added:
                self._cv.notify_all()
            return added

    def plan_merge(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        if max_batch_size <= 1:
            return []
        with self._cv:
            return self._plan_merge_locked(task_id, max_batch_size, assigned_worker)

    def _plan_merge_locked(
        self, task_id: str, max_batch_size: int, assigned_worker: str
    ) -> list[str]:
        record = self._tasks.get(task_id)
        if not record or record.status != TaskStatus.PENDING:
            return []
        if record.merge_key is None:
            return []
        if self._merge_children_map.get(task_id):
            return []
        if record.selected_worker and assigned_worker not in record.selected_worker:
            raise ValueError(
                f"The worker assigned for task {task_id} ({assigned_worker}) "
                f"is not in selected workers {record.selected_worker}."
            )
        bucket = (
            self._merge_buckets[(record.merge_key, assigned_worker)]
            + self._merge_buckets[(record.merge_key, None)]
        )
        if not bucket or len(bucket) <= 1:
            return []
        siblings: list[str] = []
        for candidate in bucket:
            if candidate == task_id:
                continue
            if len(siblings) >= max_batch_size - 1:
                break
            candidate_record = self._tasks.get(candidate)
            if not candidate_record or candidate_record.status != TaskStatus.PENDING:
                continue
            if (
                candidate_record.selected_worker
                and assigned_worker not in candidate_record.selected_worker
            ):
                continue
            if candidate not in self._ready_index:
                continue
            siblings.append(candidate)
        if not siblings:
            return []

        record.merged_children = siblings
        self._merge_children_map[task_id] = siblings.copy()
        for sibling in siblings:
            self._merge_parent_map[sibling] = task_id
            self._remove_from_ready_locked(sibling)
            self._merge_bucket_remove(sibling)
            sibling_record = self._tasks.get(sibling)
            if sibling_record:
                sibling_record.status = TaskStatus.DISPATCHED
                sibling_record.merged_parent_id = task_id
                sibling_record.assigned_worker = None
                sibling_record.merge_slice = None
        self._workflow_registry.mark_task_dispatched(record.workflow_id, *siblings)
        self._persist_locked(task_id, *siblings)

        return siblings

    def release_merge(self, task_id: str) -> None:
        with self._cv:
            self._release_merge_locked(task_id)

    def _release_merge_locked(self, task_id: str) -> None:
        children = self._merge_children_map.pop(task_id, [])
        if not children:
            parent = self._tasks.get(task_id)
            if parent:
                parent.merged_children = None
            self._persist_locked(task_id)
            return
        parent = self._tasks.get(task_id)
        if parent:
            parent.merged_children = None
        for child_id in children:
            self._merge_parent_map.pop(child_id, None)
            child_record = self._tasks.get(child_id)
            if not child_record:
                continue
            if child_record.status == TaskStatus.DONE:
                continue
            child_record.status = TaskStatus.PENDING
            child_record.merged_parent_id = None
            child_record.merge_slice = None
            if child_id not in self._ready_index:
                self._enqueue_ready_locked(child_id, front=True)
            else:
                self._remove_from_ready_locked(child_id)
                self._enqueue_ready_locked(child_id, front=True)
        self._persist_locked(task_id, *children)
        self._cv.notify_all()

    def _finalize_merged_child_success(
        self,
        child_id: str,
        worker_id: str | None,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
    ) -> list[str]:
        ready_children: list[str] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return ready_children
        child_record.status = TaskStatus.DONE
        child_record.error = None
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        if worker_id:
            child_record.assigned_worker = worker_id
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        if usage is not None:
            child_record.usages.append(usage)
        self._completed.add(child_id)
        self._failed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)
        self._workflow_registry.mark_task_done(child_record.workflow_id, child_id)
        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is None:
                continue
            pending.discard(child_id)
            if not pending:
                dep_record = self._tasks.get(dep_id)
                if dep_record and dep_record.status == TaskStatus.PENDING:
                    if self._enqueue_ready_locked(dep_id):
                        ready_children.append(dep_id)
        return ready_children

    def _finalize_merged_child_failure(
        self,
        child_id: str,
        reason: str,
        finished_ts: float,
        started_ts: float | None,
        usage: TaskUsage | None,
    ) -> list[tuple[str, str]]:
        impacted: list[tuple[str, str]] = []
        child_record = self._tasks.get(child_id)
        if not child_record:
            return impacted
        child_record.status = TaskStatus.FAILED
        child_record.error = reason
        child_record.finished_ts = finished_ts
        if started_ts is not None and child_record.started_ts is None:
            child_record.started_ts = started_ts
        child_record.assigned_worker = None
        child_record.merged_parent_id = None
        child_record.merge_slice = None
        if usage is not None:
            child_record.usages.append(usage)
        self._failed.add(child_id)
        self._completed.discard(child_id)
        self._pending_deps.pop(child_id, None)
        self._merge_parent_map.pop(child_id, None)
        self._merge_key_by_task.pop(child_id, None)
        self._remove_from_ready_locked(child_id)
        self._merge_bucket_remove(child_id)
        self._workflow_registry.mark_task_failed(child_record.workflow_id, child_id)

        dependents = list(self._dependents.pop(child_id, set()))
        for dep_id in dependents:
            pending = self._pending_deps.get(dep_id)
            if pending is not None:
                pending.discard(child_id)
            dep_record = self._tasks.get(dep_id)
            if not dep_record or dep_record.status != TaskStatus.PENDING:
                continue
            fail_reason = f"Dependency {child_id} failed"
            dep_record.status = TaskStatus.FAILED
            dep_record.error = fail_reason
            dep_record.assigned_worker = None
            dep_record.finished_ts = time.time()
            self._pending_deps.pop(dep_id, None)
            self._remove_from_ready_locked(dep_id)
            self._workflow_registry.mark_task_failed(dep_record.workflow_id, dep_id)
            impacted.append((dep_id, fail_reason))
        return impacted

    # ------------------------------------------------------------------ #
    # State updates (dispatch & events)
    # ------------------------------------------------------------------ #

    def mark_dispatched(self, task_id: str, worker: Worker) -> None:
        supplier_id = ""
        for resolver in SUPPLIER_RESOLVERS:
            if (resolved := resolver.resolve(worker)) is not None:
                supplier_id = resolved
                break

        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late dispatch must not regress a terminal task.
                return
            record.status = TaskStatus.DISPATCHED
            record.assigned_worker = worker.id
            record.topic = "tasks"
            record.dispatched_ts = time.time()
            record.next_retry_at = None
            record.supplier_id = supplier_id
            self._workflow_registry.mark_task_dispatched(record.workflow_id, task_id)
            self._remove_from_ready_locked(task_id)
            self._merge_bucket_remove(task_id)
            self._persist_locked(task_id)

    def mark_started(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
    ) -> None:
        started_ts = parse_iso_ts(str(payload.get("started_at") or ts))
        with self._cv:
            record = self._tasks.get(task_id)
            if not record:
                return
            if record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late start must not regress a terminal task.
                return
            record.status = TaskStatus.DISPATCHED
            record.started_ts = started_ts
            if worker_id:
                record.assigned_worker = worker_id
            self._workflow_registry.mark_task_dispatched(record.workflow_id, task_id)
            self._persist_locked(task_id)

    def mark_updated(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.status in TERMINAL_TASK_STATUSES:
                # A replayed or late progress update must not touch a terminal task.
                return
            record.latest_update = payload
            self._persist_locked(task_id)

    def mark_succeeded(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
    ) -> list[tuple[str, TaskUsage]]:
        """
        Mark a task as completed and enqueue any dependents that have become ready.
        Returns the per-task usage rows produced by the completion.
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        # TODO(kaiitunnz): Make usage task-specific
        usage = TaskUsage.from_payload(payload, TaskStatus.DONE)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return usages
                if record.status == TaskStatus.DONE:
                    # Idempotent: a replayed TASK_SUCCEEDED must not re-apply.
                    return []
                record.status = TaskStatus.DONE
                record.error = None
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None
                if usage is not None:
                    record.usages.append(usage)
                self._workflow_registry.mark_task_done(record.workflow_id, task_id)

            self._completed.add(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            ready_children: list[str] = []
            merged_children_ids: list[str] = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is None:
                    continue
                pending.discard(task_id)
                if not pending:
                    child_record = self._tasks.get(child)
                    if child_record and child_record.status == TaskStatus.PENDING:
                        if self._enqueue_ready_locked(child):
                            ready_children.append(child)

            for merged_child in merged_children_ids:
                ready_children.extend(
                    self._finalize_merged_child_success(
                        merged_child,
                        worker_id,
                        finished_ts,
                        started_ts,
                        usage,
                    )
                )

            if record is not None:
                ready_children.extend(
                    self._try_advance_epoch_frontier_locked(record.workflow_id)
                )

            self._persist_locked(task_id, *merged_children_ids)
            if record is not None:
                self._persist_sched_locked(record.workflow_id)

            if ready_children:
                self._cv.notify_all()

            return usages

    def mark_failed(
        self,
        task_id: str,
        worker_id: str | None,
        payload: dict[str, Any],
        ts: str,
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[str, str]], list[str], list[tuple[str, TaskUsage]]]:
        """
        Mark a task as failed. Dependent tasks still waiting on this task are
        automatically failed to avoid running without prerequisites.

        Returns (impacted_dependents, merged_children_ids, usages).
        """
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        message = error or str(payload.get("error") or "task failed")
        # TODO(kaiitunnz): Make usage task-specific
        usage = TaskUsage.from_payload(payload, TaskStatus.FAILED)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record:
                if record.status == TaskStatus.CANCELLED:
                    return [], [], usages
                if record.status == TaskStatus.FAILED:
                    # Idempotent: a replayed TASK_FAILED must not re-apply.
                    return [], [], []
                record.status = TaskStatus.FAILED
                record.error = message
                record.finished_ts = finished_ts
                if started_ts:
                    record.started_ts = started_ts
                if worker_id:
                    record.assigned_worker = worker_id
                record.merged_children = None
                if usage is not None:
                    record.usages.append(usage)
                self._workflow_registry.mark_task_failed(record.workflow_id, task_id)

            self._failed.add(task_id)
            self._completed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            merged_children_ids = self._merge_children_map.pop(task_id, [])
            self._merge_key_by_task.pop(task_id, None)

            impacted: list[tuple[str, str]] = []
            dependents = list(self._dependents.pop(task_id, set()))
            for child in dependents:
                pending = self._pending_deps.get(child)
                if pending is not None:
                    pending.discard(task_id)
                child_record = self._tasks.get(child)
                if not child_record or child_record.status != TaskStatus.PENDING:
                    continue
                reason = f"Dependency {task_id} failed"
                child_record.status = TaskStatus.FAILED
                child_record.error = reason
                child_record.assigned_worker = None
                child_record.finished_ts = time.time()
                self._pending_deps.pop(child, None)
                self._remove_from_ready_locked(child)
                self._workflow_registry.mark_task_failed(
                    child_record.workflow_id, child
                )
                impacted.append((child, reason))

            for merged_child in merged_children_ids:
                impacted.extend(
                    self._finalize_merged_child_failure(
                        merged_child,
                        f"Parent {task_id} failed",
                        finished_ts,
                        started_ts,
                        usage,
                    )
                )

            failed_epoch = self._task_epoch_index.get(task_id)
            if record and failed_epoch is not None:
                impacted.extend(
                    self._fail_later_epochs_locked(
                        record.workflow_id,
                        failed_epoch,
                        f"Blocked by failed task {task_id} in earlier epoch",
                    )
                )

            self._persist_locked(
                task_id, *merged_children_ids, *(dep_id for dep_id, _ in impacted)
            )
            if record is not None:
                self._persist_sched_locked(record.workflow_id)

            return impacted, merged_children_ids, usages

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def cancel_workflow(self, workflow_id: str, reason: str = "cancelled") -> list[str]:
        cancelled: list[str] = []
        interrupts: list[InterruptMessage] = []
        with self._cv:
            workflow_task_ids = [
                task_id
                for task_id, record in self._tasks.items()
                if record.workflow_id == workflow_id
            ]
            for task_id, record in list(self._tasks.items()):
                if record.workflow_id != workflow_id:
                    continue
                match record.status:
                    case TaskStatus.PENDING if not self._parent_is_active(task_id):
                        record.status = TaskStatus.CANCELLED
                        record.error = reason
                        record.finished_ts = time.time()
                        record.assigned_worker = None
                        record.merged_children = None
                        # TODO(kaiitunnz): Handle usages for cancelled tasks
                        self._completed.discard(task_id)
                        self._failed.discard(task_id)
                        self._pending_deps.pop(task_id, None)
                        self._remove_from_ready_locked(task_id)
                        self._merge_bucket_remove(task_id)
                        self._merge_key_by_task.pop(task_id, None)
                        self._merge_parent_map.pop(task_id, None)
                        self._merge_children_map.pop(task_id, None)
                    case TaskStatus.DISPATCHED if (
                        not self._parent_is_active(task_id)
                        and not record.merged_children
                        and record.assigned_worker
                    ):
                        record.status = TaskStatus.CANCELLING
                        record.error = reason
                        interrupts.append(
                            InterruptMessage(
                                task_id=task_id,
                                worker_id=record.assigned_worker,
                                reason=reason,
                            )
                        )
                    case _:
                        continue
                cancelled.append(task_id)

            self._workflow_epoch_tasks.pop(workflow_id, None)
            self._workflow_epoch_frontier.pop(workflow_id, None)
            self._workflow_in_epoch_order.pop(workflow_id, None)
            for task_id in workflow_task_ids:
                self._task_epoch_index.pop(task_id, None)
            self._persist_locked(*cancelled)
            self._persist_sched_locked(workflow_id)

        for task_id in cancelled:
            maybe_record = self._tasks.get(task_id)
            if maybe_record and maybe_record.status == TaskStatus.CANCELLED:
                self._workflow_registry.mark_task_cancelled(workflow_id, task_id)
        for interrupt in interrupts:
            worker = self._worker_registry.get_worker(interrupt.worker_id)
            if worker is None:
                self._logger.warning(
                    "Cannot publish interrupt for %s; worker %s missing",
                    interrupt.task_id,
                    interrupt.worker_id,
                )
            else:
                self._worker_registry.publish_interrupt(worker, interrupt)
        return cancelled

    def mark_cancelled(
        self, task_id: str, worker_id: str | None, payload: dict[str, Any], ts: str
    ) -> list[tuple[str, TaskUsage]]:
        finished_ts = parse_iso_ts(str(payload.get("finished_at") or ts))
        maybe_started = payload.get("started_at")
        started_ts = parse_iso_ts(str(maybe_started)) if maybe_started else None
        usage = TaskUsage.from_payload(payload, TaskStatus.CANCELLED)
        usages: list[tuple[str, TaskUsage]] = []
        if usage is not None:
            usages.append((task_id, usage))

        with self._cv:
            record = self._tasks.get(task_id)
            if record is None:
                return usages
            if record.status == TaskStatus.CANCELLED:
                return usages
            record.status = TaskStatus.CANCELLED
            record.finished_ts = finished_ts
            if started_ts:
                record.started_ts = started_ts
            record.merged_children = None
            if usage is not None:
                record.usages.append(usage)
            # TODO(kaiitunnz): Handle usages for cancelled tasks
            self._completed.discard(task_id)
            self._failed.discard(task_id)
            self._pending_deps.pop(task_id, None)
            self._remove_from_ready_locked(task_id)
            self._merge_bucket_remove(task_id)
            self._merge_key_by_task.pop(task_id, None)
            self._merge_children_map.pop(task_id, None)
            self._workflow_registry.mark_task_cancelled(record.workflow_id, task_id)
            record.assigned_worker = None
            self._persist_locked(task_id)
            return usages

    def get_record(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_merged_children(self, task_id: str) -> list[str]:
        """Read the merged-children list without consuming it."""
        with self._cv:
            return self._merge_children_map.get(task_id, [])

    def describe_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            return self._build_task_info_locked(task_id, record)

    def list_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return [
                self._build_task_info_locked(task_id, record)
                for task_id, record in self._tasks.items()
            ]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @property
    def tasks(self) -> dict[str, TaskRecord]:
        return self._tasks

    def recover_tasks_for_worker(self, worker_id: str) -> list[str]:
        """
        Move DISPATCHED tasks assigned to a departed worker back to the ready queue.
        Returns affected task_ids.
        """
        recovered: list[str] = []
        with self._cv:
            for task_id, record in self._tasks.items():
                if record.assigned_worker != worker_id:
                    continue
                if record.status not in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    continue
                self._rehydrated_dispatched.pop(task_id, None)
                recovered.append(task_id)
        return recovered

    def has_rehydrated_in_flight(self, worker_id: str, within_sec: float) -> bool:
        """
        Whether ``worker_id`` still owns an in-flight task that was rehydrated
        within the last ``within_sec`` seconds.

        Worker heartbeats are dropped while the root is down, so a surviving
        worker looks briefly stale right after a restart. The watchdog uses this
        to extend a worker's death grace until its rehydrated tasks' window has
        elapsed, giving the worker time to re-register before its tasks are
        reclaimed.
        """
        now = time.time()
        with self._cv:
            for task_id, rehydrated_at in list(self._rehydrated_dispatched.items()):
                record = self._tasks.get(task_id)
                if record is None or record.status not in (
                    TaskStatus.DISPATCHED,
                    TaskStatus.CANCELLING,
                ):
                    self._rehydrated_dispatched.pop(task_id, None)
                    continue
                if now - rehydrated_at >= within_sec:
                    continue
                if record.assigned_worker == worker_id:
                    return True
        return False

    def shutdown(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def ready_queue_length(self) -> int:
        with self._cv:
            return len(self._ready_queue)

    def queued_gpu_counts(self) -> set[int]:
        """Return the set of distinct GPU counts requested by tasks in the ready queue.

        0 represents a CPU-only task.  Used to match each candidate server to
        the best worker it can create for the current queue.
        """
        counts: set[int] = set()
        with self._cv:
            for task_id, _ in self._ready_queue:
                record = self._tasks.get(task_id)
                if record is None:
                    continue
                resources = record.task.spec.resources
                if resources is None or resources.hardware is None:
                    counts.add(0)
                    continue
                gpu = resources.hardware.gpu
                if gpu:
                    # Default to 1 if a GPU is required but count is unspecified
                    counts.add(int(gpu.count) if gpu.count else 1)
                else:
                    counts.add(0)
        return counts

    def task_status_counts(self) -> tuple[int, int, int, int, int]:
        with self._cv:
            queueing = len(self._ready_queue)
            dispatched = 0
            pending = 0
            done = 0
            for task_id, record in self._tasks.items():
                status = record.status
                if status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING):
                    dispatched += 1
                elif status == TaskStatus.DONE:
                    done += 1
                elif status == TaskStatus.PENDING and task_id not in self._ready_index:
                    pending += 1
            total = len(self._tasks)
            return queueing, dispatched, pending, done, total

    def _build_task_info_locked(self, task_id: str, record: TaskRecord) -> TaskInfo:
        return TaskInfo(
            **dict(record),
            depends_on=sorted(self._original_deps.get(task_id, set())),
            pending_dependencies=sorted(self._pending_deps.get(task_id, set())),
            dependents=sorted(self._dependents.get(task_id, set())),
            completed=task_id in self._completed,
            failed=task_id in self._failed,
        )

    def _parent_is_active(self, task_id: str) -> bool:
        parent_id = self._merge_parent_map.get(task_id)
        if not parent_id:
            return False
        parent_record = self._tasks.get(parent_id)
        if not parent_record:
            return False
        return parent_record.status in (TaskStatus.DISPATCHED, TaskStatus.CANCELLING)
