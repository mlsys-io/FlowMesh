import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from shared.schemas.result import result_file_path
from shared.utils.manifest import sync_manifest

from ..clients.redis import (
    TASK_LOGS_STREAM_PREFIX,
    SyncRedisClient,
    task_log_archive_last_id_key,
    task_log_stream_key,
)
from ..task.models import TaskStatus
from ..task.runtime import TaskRuntime


@dataclass(slots=True)
class _TaskArchiveState:
    last_id: str
    last_flush_ts: float
    done: bool


class TaskLogArchiver:
    def __init__(
        self,
        redis: SyncRedisClient,
        runtime: TaskRuntime,
        results_dir: Path,
        logger: logging.Logger,
        flush_interval_sec: float = 5.0,
        flush_max_entries: int = 100,
    ) -> None:
        self._redis = redis
        self._runtime = runtime
        self._results_dir = results_dir
        self._logger = logger
        self._flush_interval_sec = max(0.1, float(flush_interval_sec))
        self._flush_max_entries = max(1, int(flush_max_entries))

        self._states: dict[str, _TaskArchiveState] = {}
        """task_id -> _TaskArchiveState"""
        self._buffers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        """task_id -> list of (stream_id, fields)"""

    def run(self, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self._logger.debug("Log archiver tick failed: %s", exc)
                time.sleep(1.0)

    def _tick(self) -> None:
        now = time.time()
        tasks = self._runtime.list_tasks()
        terminal: set[str] = set()

        # Ensure all tasks are being tracked
        for task in tasks:
            task_id = task.task_id
            if task.status in {
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                if (
                    self._load_checkpoint(task_id) is None
                    and self._logs_path(task_id).exists()
                ):
                    continue
                terminal.add(task_id)
            self._ensure_task(task_id, now)

        active = [task_id for task_id, state in self._states.items() if not state.done]
        if not active:
            time.sleep(0.5)
            return

        streams: dict[bytes | str | memoryview, int | bytes | str | memoryview] = {}
        for task_id in active:
            key = task_log_stream_key(task_id)
            streams[key] = self._states[task_id].last_id

        # Read new log entries
        rows = self._redis.xread_telemetry(streams, count=500, block_ms=1000)
        for stream_key, batch in rows:
            task_id = stream_key.removeprefix(TASK_LOGS_STREAM_PREFIX)
            buf = self._buffers.setdefault(task_id, [])
            for stream_id, fields in batch:
                buf.append((stream_id, fields))
                self._states[task_id].last_id = stream_id

        # Flush buffers
        for task_id in active:
            buffer = self._buffers.get(task_id) or []
            state = self._states[task_id]
            should_flush = len(buffer) >= self._flush_max_entries or (
                buffer and (now - state.last_flush_ts) >= self._flush_interval_sec
            )
            if should_flush:
                self._flush_task(task_id, buffer)
                self._buffers[task_id] = []
                state.last_flush_ts = now

        # Finalize terminal tasks
        for task_id in terminal:
            maybe_state = self._states.get(task_id)
            if not maybe_state or maybe_state.done:
                continue
            try:
                self._drain_task(task_id)
                self._finalize_manifest(task_id)
                maybe_state.done = True
            finally:
                self._buffers.pop(task_id, None)
                self._states.pop(task_id, None)

    def _ensure_task(self, task_id: str, now: float) -> None:
        if task_id in self._states:
            return
        last_id = self._load_checkpoint(task_id) or "0-0"
        self._states[task_id] = _TaskArchiveState(
            last_id=last_id, last_flush_ts=now, done=False
        )
        self._buffers.setdefault(task_id, [])

    def _task_logs_dir(self, task_id: str) -> Path:
        base_dir = result_file_path(self._results_dir, task_id).parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _logs_path(self, task_id: str) -> Path:
        return self._task_logs_dir(task_id) / "logs.jsonl"

    def _load_checkpoint(self, task_id: str) -> str | None:
        last_id = self._redis.get(task_log_archive_last_id_key(task_id))
        return last_id or None

    def _save_checkpoint(self, task_id: str, last_id: str) -> None:
        self._redis.set_value(task_log_archive_last_id_key(task_id), last_id)

    def _flush_task(
        self, task_id: str, items: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not items:
            return
        logs_path = self._logs_path(task_id)
        last_id = self._states[task_id].last_id
        with logs_path.open("a", encoding="utf-8") as fh:
            for _, fields in items:
                payload = fields.get("payload")
                if not isinstance(payload, str) or not payload:
                    continue
                try:
                    json.loads(payload)
                    fh.write(payload + "\n")
                except json.JSONDecodeError:
                    wrapper = {
                        "message": payload,
                        "level": "INFO",
                        "stream": "system",
                    }
                    fh.write(json.dumps(wrapper, ensure_ascii=False) + "\n")
        self._save_checkpoint(task_id, last_id)

    def _drain_task(self, task_id: str) -> None:
        state = self._states[task_id]
        start = state.last_id
        while True:
            key = task_log_stream_key(task_id)
            batch = self._redis.xrange_telemetry(key, min_id=f"({start}", count=1000)
            if not batch:
                return
            self._buffers.setdefault(task_id, []).extend(batch)
            start = batch[-1][0]
            state.last_id = start
            if len(self._buffers[task_id]) >= self._flush_max_entries:
                self._flush_task(task_id, self._buffers[task_id])
                self._buffers[task_id] = []

    def _finalize_manifest(self, task_id: str) -> None:
        record = self._runtime.get_record(task_id)
        expected_artifacts: list[str] = []
        if record:
            expected_artifacts = record.task.spec.get_artifacts()
        expected_artifacts.append("logs/logs.jsonl")
        base_dir = result_file_path(self._results_dir, task_id).parent
        self._logs_path(task_id).touch(exist_ok=True)
        try:
            sync_manifest(base_dir, task_id, expected_artifacts)
        except Exception as exc:
            self._logger.debug("Failed to sync manifest for %s: %s", task_id, exc)
        try:
            self._redis.delete(task_log_archive_last_id_key(task_id))
        except Exception as exc:
            self._logger.debug(
                "Failed to delete archive checkpoint for %s: %s", task_id, exc
            )
