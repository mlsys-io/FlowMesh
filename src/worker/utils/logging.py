import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import grpc

from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc
from shared.utils.time import now_iso


def get_logger(
    name: str = "flowmesh_worker",
    log_file: str = "worker.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """Return a configured logger with a rotating file handler and console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # File handler (rotating)
    fh = RotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger


def configure_hf_library_logging() -> None:
    """Disable HF default handlers and enable propagation to root.

    Safe to call multiple times; no-ops if the libraries are not installed.
    """

    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:
        pass

    try:
        from diffusers.utils import logging as diff_logging  # type: ignore

        diff_logging.disable_default_handler()
        diff_logging.enable_propagation()
    except Exception:
        pass


class _GrpcLogStream:
    _SENTINEL = object()

    def __init__(
        self,
        stub: supervisor_pb2_grpc.SupervisorStub,
        metadata: tuple[tuple[str, str], ...],
        struct_from_payload: Callable[[dict[str, Any]], Any],
        logger: logging.Logger,
    ) -> None:
        self._stub = stub
        self._metadata = metadata
        self._struct_from_payload = struct_from_payload
        self._logger = logger

        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=10_000)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ServerLogStream",
            daemon=True,
        )
        self._thread.start()

    def send(self, payload: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            self._q.put(payload, timeout=0.1)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put(self._SENTINEL)
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _messages(self) -> Iterable[supervisor_pb2.LogMessage]:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            if isinstance(item, dict):
                yield supervisor_pb2.LogMessage(payload=self._struct_from_payload(item))

    def _run(self) -> None:
        try:
            self._stub.PushLogs(self._messages(), metadata=self._metadata)
        except grpc.RpcError as exc:
            self._logger.debug("Server log stream error: %s", exc)
        except Exception as exc:
            self._logger.debug("Server log stream crashed: %s", exc)


class _JsonlLogSink:
    _SENTINEL = object()

    def __init__(
        self,
        log_paths: dict[str, Path],
        logger: logging.Logger,
        flush_interval_sec: float = 5.0,
        flush_max_entries: int = 100,
    ) -> None:
        self._log_paths = log_paths
        self._logger = logger
        self._flush_interval_sec = max(0.1, float(flush_interval_sec))
        self._flush_max_entries = max(1, int(flush_max_entries))

        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=10_000)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TaskJsonlLogSink",
            daemon=True,
        )
        self._files: dict[str, TextIOWrapper] = {}
        self._buffer: list[dict[str, Any]] = []
        self._thread.start()

    def send(self, payload: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            self._q.put(payload, timeout=0.1)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put(self._SENTINEL)
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _ensure_handle(self, task_id: str) -> TextIOWrapper | None:
        handle = self._files.get(task_id)
        if handle is not None:
            return handle
        path = self._log_paths.get(task_id)
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
        except Exception:
            self._logger.debug(
                "Failed to open task log file for %s at %s", task_id, path
            )
            return None
        self._files[task_id] = handle
        return handle

    def _flush(self) -> None:
        if not self._buffer:
            return
        buffered = self._buffer
        self._buffer = []
        try:
            for payload in buffered:
                line = json.dumps(payload, ensure_ascii=False)
                task_refs = payload["task_refs"]
                for ref in task_refs:
                    task_id = ref["task_id"]
                    handle = self._ensure_handle(task_id)
                    if handle is None:
                        continue
                    try:
                        handle.write(line + "\n")
                    except Exception:
                        continue
            for handle in self._files.values():
                try:
                    handle.flush()
                except Exception:
                    pass
        except Exception:
            self._logger.debug("Task JSONL log sink flush failed", exc_info=True)

    def _run(self) -> None:
        last_flush = 0.0
        try:
            last_flush = time.time()
            while True:
                timeout = max(0.1, self._flush_interval_sec / 2.0)
                try:
                    item = self._q.get(timeout=timeout)
                except queue.Empty:
                    item = None
                if item is self._SENTINEL:
                    break
                now = time.time()
                if isinstance(item, dict):
                    self._buffer.append(item)
                if (len(self._buffer) >= self._flush_max_entries) or (
                    now - last_flush >= self._flush_interval_sec
                ):
                    self._flush()
                    last_flush = now
        finally:
            self._flush()
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass


class TaskLogEmitter(logging.Handler):
    """Per-task log handler that emits Python logging records to the server."""

    _traceback_formatter = logging.Formatter()

    def __init__(
        self,
        stub: supervisor_pb2_grpc.SupervisorStub,
        metadata: tuple[tuple[str, str], ...],
        struct_from_payload: Callable[[dict[str, Any]], Any],
        logger: logging.Logger,
        task_id: str,
        workflow_id: str,
        owner_id: str,
        worker_id: str,
        task_refs: list[dict[str, str]] | None = None,
        log_paths: dict[str, Path] | None = None,
        flush_interval_sec: float = 5.0,
        flush_max_entries: int = 100,
    ) -> None:
        super().__init__(level=logging.NOTSET)
        self._logger = logger
        self._task_id = task_id
        self._workflow_id = workflow_id
        self._owner_id = owner_id
        self._worker_id = worker_id
        self._task_refs = (
            [{"task_id": task_id, "workflow_id": workflow_id}]
            if task_refs is None
            else task_refs
        )
        self._stream = _GrpcLogStream(
            stub=stub,
            metadata=metadata,
            struct_from_payload=struct_from_payload,
            logger=logger,
        )
        self._sink: _JsonlLogSink | None = None
        if log_paths:
            self._sink = _JsonlLogSink(
                log_paths=log_paths,
                logger=logger,
                flush_interval_sec=flush_interval_sec,
                flush_max_entries=flush_max_entries,
            )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", ""))
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self._traceback_formatter.formatException(
                    record.exc_info
                )
            if exc_text := record.exc_text:
                message = f"{message}\n{exc_text}" if message else exc_text
        if not message:
            return

        stream = getattr(record, "flowmesh_stream", None)
        if stream not in ("stdout", "stderr", "system"):
            stream = "system"

        payload: dict[str, Any] = {
            "type": "TASK_LOG",
            "ts": now_iso(),
            "workflow_id": self._workflow_id,
            "task_id": self._task_id,
            "task_refs": self._task_refs,
            "owner_id": self._owner_id,
            "worker_id": self._worker_id,
            "level": record.levelname,
            "stream": stream,
            "logger": record.name,
            "message": message,
        }
        self._stream.send(payload)
        if self._sink is not None:
            self._sink.send(payload)

    def close(self) -> None:
        try:
            if self._sink is not None:
                self._sink.close()
            self._stream.close()
        finally:
            super().close()

    def emit_warning_only(self, message: str) -> None:
        """Send a single warning log line and do not attach the handler."""
        payload: dict[str, Any] = {
            "type": "TASK_LOG",
            "ts": now_iso(),
            "workflow_id": self._workflow_id,
            "task_id": self._task_id,
            "task_refs": self._task_refs,
            "owner_id": self._owner_id,
            "worker_id": self._worker_id,
            "level": "WARNING",
            "stream": "system",
            "logger": "task_log_emitter",
            "message": message,
        }
        try:
            self._stream.send(payload)
            if self._sink is not None:
                self._sink.send(payload)
        finally:
            if self._sink is not None:
                self._sink.close()
            self._stream.close()
