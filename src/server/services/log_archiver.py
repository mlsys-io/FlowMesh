import json
import logging
import os
import stat
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
    task_log_archived_key,
    task_log_stream_key,
)
from ..task.models import TaskStatus
from ..task.runtime import TaskRuntime

# A failed flush is retried after a delay that doubles from the first, up to the flush
# interval. A task whose writes keep failing for the give-up window has its buffer
# dropped: a volume that fills briefly, as when another task's output is cleaned up,
# loses nothing, while a broken one cannot hold lines forever. A retrying task reads
# no new lines, so its buffer holds at most what it had read when the writes failed,
# and a tick with only retrying tasks waits for the earliest retry, at most as long
# as a stream read blocks.
_FIRST_RETRY_SEC = 1.0
_GIVE_UP_SEC = 300.0
_READ_BLOCK_SEC = 1.0


class _TornWrite(OSError):
    """A failed append whose partial write could not be truncated away."""


@dataclass(slots=True)
class _TaskArchiveState:
    last_id: str
    last_flush_ts: float
    done: bool
    failures: int = 0
    first_failure_ts: float | None = None
    next_attempt_ts: float = 0.0

    @property
    def retrying(self) -> bool:
        return self.first_failure_ts is not None


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
        self._archived_ids: set[str] = set()
        """Finished tasks known to need no archiving, so each is probed once."""

    def run(self, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self._logger.debug("Log archiver tick failed: %s", exc)
                time.sleep(1.0)

    def _tick(self) -> None:
        now = time.time()
        terminal: set[str] = set()

        # Ensure all tasks are being tracked
        statuses = self._runtime.task_statuses()
        self._archived_ids.intersection_update(statuses)
        for task_id, task_status in statuses.items():
            if task_status in {
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                if task_id in self._archived_ids:
                    continue
                if task_id not in self._states:
                    archived = self._archived(task_id)
                    if archived is None:
                        continue
                    if archived:
                        self._archived_ids.add(task_id)
                        continue
                terminal.add(task_id)
            self._ensure_task(task_id, now)

        active = [task_id for task_id, state in self._states.items() if not state.done]
        if not active:
            time.sleep(0.5)
            return

        streams: dict[bytes | str | memoryview, int | bytes | str | memoryview] = {}
        for task_id in active:
            if not self._states[task_id].retrying:
                key = task_log_stream_key(task_id)
                streams[key] = self._states[task_id].last_id

        # Read new log entries
        rows: list[Any] = []
        if streams:
            rows = self._redis.xread_telemetry(
                streams, count=500, block_ms=int(_READ_BLOCK_SEC * 1000)
            )
        else:
            retry_ts = min(self._states[task_id].next_attempt_ts for task_id in active)
            if retry_ts > now:
                time.sleep(min(retry_ts - now, _READ_BLOCK_SEC))
        now = time.time()
        for stream_key, batch in rows:
            task_id = stream_key.removeprefix(TASK_LOGS_STREAM_PREFIX)
            buf = self._buffers.setdefault(task_id, [])
            for stream_id, fields in batch:
                buf.append((stream_id, fields))
                self._states[task_id].last_id = stream_id

        # Flush buffers; a terminal task's are flushed as it is drained below
        for task_id in active:
            buffer = self._buffers.get(task_id) or []
            state = self._states[task_id]
            if task_id in terminal or now < state.next_attempt_ts:
                continue
            should_flush = (
                state.retrying
                or len(buffer) >= self._flush_max_entries
                or (buffer and (now - state.last_flush_ts) >= self._flush_interval_sec)
            )
            if should_flush:
                self._flush_buffer(task_id, now)
                state.last_flush_ts = now

        # Finalize terminal tasks once their last lines are written
        for task_id in terminal:
            maybe_state = self._states.get(task_id)
            if not maybe_state or maybe_state.done:
                continue
            if now < maybe_state.next_attempt_ts:
                continue
            try:
                if not self._drain_task(task_id, now):
                    continue
                self._finalize_manifest(task_id)
                maybe_state.done = True
                self._archived_ids.add(task_id)
            except Exception:
                self._forget(task_id)
                raise
            self._forget(task_id)

    def _forget(self, task_id: str) -> None:
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

    def _base_dir(self, task_id: str) -> Path:
        return result_file_path(self._results_dir, task_id).parent

    def _task_logs_dir(self, task_id: str) -> Path:
        base_dir = result_file_path(self._results_dir, task_id).parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _logs_path(self, task_id: str) -> Path:
        return self._task_logs_dir(task_id) / "logs.jsonl"

    def _archived(self, task_id: str) -> bool | None:
        """Whether a finished task's logs need no archiving: it was finalized, or,
        finalized before that was recorded, its log file holds lines or something
        other than a file stands where it or a directory holding it belongs. None
        when its log file could not be checked."""
        checkpoint = self._redis.get(task_log_archived_key(task_id))
        if checkpoint:
            return True
        if self._redis.get(task_log_archive_last_id_key(task_id)):
            return False
        try:
            st = os.stat(self._logs_path(task_id), follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as exc:
            self._logger.debug("Could not check %s's log file: %s", task_id, exc)
            return None
        return not stat.S_ISREG(st.st_mode) or st.st_size > 0

    def _load_checkpoint(self, task_id: str) -> str | None:
        return self._redis.get(task_log_archive_last_id_key(task_id)) or None

    def _save_checkpoint(self, task_id: str, last_id: str) -> None:
        self._redis.set_value(task_log_archive_last_id_key(task_id), last_id)

    def _flush_buffer(self, task_id: str, now: float) -> bool:
        """Flush the task's buffer; return whether it is done with, written or
        dropped, rather than kept for a retry."""
        if not (buffer := self._buffers.get(task_id)):
            return True
        if not self._flush_task(task_id, buffer, now):
            return False
        self._buffers[task_id] = []
        return True

    def _flush_task(
        self, task_id: str, items: list[tuple[str, dict[str, Any]]], now: float
    ) -> bool:
        """Append ``items`` to the task's log file; return whether they are done
        with, written or dropped, rather than kept for a retry."""
        if not items:
            return True
        state = self._states[task_id]
        last_id = state.last_id
        lines: list[str] = []
        for _, fields in items:
            payload = fields.get("payload")
            if not isinstance(payload, str) or not payload:
                continue
            try:
                json.loads(payload)
                lines.append(payload)
            except json.JSONDecodeError:
                wrapper = {"message": payload, "level": "INFO", "stream": "system"}
                lines.append(json.dumps(wrapper, ensure_ascii=False))
        data = "".join(f"{line}\n" for line in lines).encode("utf-8")
        try:
            self._append(task_id, data)
        except _TornWrite as exc:
            self._logger.error(
                "Dropping %d log lines for %s: %s", len(lines), task_id, exc
            )
        except OSError as exc:
            if state.first_failure_ts is None:
                state.first_failure_ts = now
            if now - state.first_failure_ts < _GIVE_UP_SEC:
                delay = min(
                    _FIRST_RETRY_SEC * 2**state.failures, self._flush_interval_sec
                )
                state.failures += 1
                state.next_attempt_ts = now + delay
                self._logger.warning(
                    "Archiving logs for %s failed, retrying in %.0f s: %s",
                    task_id,
                    delay,
                    exc,
                )
                return False
            self._logger.error(
                "Dropping %d log lines for %s after %.0f s of failed writes: %s",
                len(lines),
                task_id,
                now - state.first_failure_ts,
                exc,
            )
        state.failures = 0
        state.first_failure_ts = None
        state.next_attempt_ts = 0.0
        self._save_checkpoint(task_id, last_id)
        return True

    def _append(self, task_id: str, data: bytes) -> None:
        """Append ``data`` to the task's log file whole or not at all: a failed write
        is truncated back to the file's prior end, unless another writer appended
        meanwhile, whose lines the truncate would cut."""
        logs_path = self._logs_path(task_id)
        with logs_path.open("a", encoding="utf-8") as fh:
            fd = fh.fileno()
            end = os.fstat(fd).st_size
            written = 0
            try:
                while written < len(data):
                    written += os.write(fd, data[written:])
            except OSError as exc:
                if written:
                    self._undo_append(fd, end, written, exc)
                raise

    @staticmethod
    def _undo_append(fd: int, end: int, written: int, exc: OSError) -> None:
        try:
            appended = os.fstat(fd).st_size != end + written
            if not appended:
                os.ftruncate(fd, end)
        except OSError as truncate_exc:
            raise _TornWrite(
                truncate_exc.errno,
                f"a failed write ({exc}) could not be truncated away",
            ) from exc
        if appended:
            raise _TornWrite(
                exc.errno, f"another writer appended beside a failed write ({exc})"
            ) from exc

    def _drain_task(self, task_id: str, now: float) -> bool:
        """Read and write the rest of the task's log stream; return whether every
        line is done with, written or dropped. Lines already read are written before
        any more are read."""
        if not self._flush_buffer(task_id, now):
            return False
        state = self._states[task_id]
        start = state.last_id
        while True:
            key = task_log_stream_key(task_id)
            batch = self._redis.xrange_telemetry(key, min_id=f"({start}", count=1000)
            if not batch:
                return self._flush_buffer(task_id, now)
            self._buffers.setdefault(task_id, []).extend(batch)
            start = batch[-1][0]
            state.last_id = start
            if len(self._buffers[task_id]) >= self._flush_max_entries:
                if not self._flush_buffer(task_id, now):
                    return False

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
            self._redis.set_value(task_log_archived_key(task_id), "1")
            self._redis.delete(task_log_archive_last_id_key(task_id))
        except Exception as exc:
            self._logger.debug(
                "Failed to record %s's logs as archived: %s", task_id, exc
            )
