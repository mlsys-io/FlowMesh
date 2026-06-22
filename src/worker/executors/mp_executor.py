"""Utility wrapper that executes another Executor in an isolated process.

This is primarily meant for CUDA-related executors (e.g., vLLM) where GPU
contexts are only fully released once the owning process exits. Instead
of keeping those executors resident in the main worker process, we spin
up a short-lived subprocess per task, run the real executor there, and
forward the result/exception back to the parent.
"""

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import psutil

from shared.schemas.result import BaseExecutorResult
from shared.tasks.worker_message import WorkerHardware
from worker.config import WorkerConfig

from .base_executor import ExecutionError, Executor, ExecutorTask

logger = logging.getLogger(__name__)

_RESULT_POLL_INTERVAL_SEC = 2.0


class MPLogHandler:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._og_stdout_fd: int | None = None
        self._og_stderr_fd: int | None = None
        self._out_thread: threading.Thread | None = None
        self._err_thread: threading.Thread | None = None

    def __enter__(self) -> "MPLogHandler":
        """Redirect stdout/stderr to pipes and forward them into logging."""
        if not self._enabled:
            return self
        # Duplicate original fds so we can restore them.
        self._og_stdout_fd = os.dup(1)
        self._og_stderr_fd = os.dup(2)

        # Create pipes and redirect process fds.
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        os.dup2(stdout_w, 1)
        os.dup2(stderr_w, 2)
        os.close(stdout_w)
        os.close(stderr_w)

        out_logger = logging.getLogger("worker.subprocess.stdout")
        err_logger = logging.getLogger("worker.subprocess.stderr")
        self._out_thread = self._forward_fd_lines_to_logger(
            stdout_r, out_logger, logging.INFO, stream="stdout"
        )
        self._err_thread = self._forward_fd_lines_to_logger(
            stderr_r, err_logger, logging.ERROR, stream="stderr"
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Restore original stdout/stderr fds and wait for forwarder threads."""
        if not self._enabled:
            return
        if self._og_stdout_fd is not None and self._og_stderr_fd is not None:
            try:
                os.dup2(self._og_stdout_fd, 1)
            except Exception:
                pass
            try:
                os.dup2(self._og_stderr_fd, 2)
            except Exception:
                pass
            for fd in (self._og_stdout_fd, self._og_stderr_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass
            self._og_stdout_fd = None
            self._og_stderr_fd = None
        if self._out_thread is not None:
            try:
                self._out_thread.join(timeout=0.2)
            except Exception:
                pass
            self._out_thread = None
        if self._err_thread is not None:
            try:
                self._err_thread.join(timeout=0.2)
            except Exception:
                pass
            self._err_thread = None

    def _forward_fd_lines_to_logger(
        self, fd: int, target_logger: logging.Logger, level: int, *, stream: str
    ) -> threading.Thread:
        """Forward a file descriptor's lines into Python logging."""

        def _run() -> None:
            try:
                with os.fdopen(fd, "rb", closefd=True) as fh:
                    buf = b""
                    while True:
                        chunk = fh.read(4096)
                        if not chunk:
                            break
                        # Some writers (e.g. tqdm/progress bars) emit carriage returns
                        # without newlines. Normalize to newlines to avoid unbounded
                        # buffering.
                        if b"\r" in chunk:
                            chunk = chunk.replace(b"\r", b"\n")
                        buf += chunk
                        if len(buf) > 65_536:
                            text = buf.decode("utf-8", errors="replace")
                            text = text.rstrip("\n")
                            inferred = self._infer_level_from_line(text, default=level)
                            target_logger.log(
                                inferred,
                                text,
                                extra={"flowmesh_stream": stream},
                            )
                            buf = b""
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            text = line.decode("utf-8", errors="replace").rstrip("\r")
                            if not text:
                                continue
                            inferred = self._infer_level_from_line(text, default=level)
                            target_logger.log(
                                inferred,
                                text,
                                extra={"flowmesh_stream": stream},
                            )
                    if buf:
                        text = buf.decode("utf-8", errors="replace").rstrip("\r")
                        if text:
                            inferred = self._infer_level_from_line(text, default=level)
                            target_logger.log(
                                inferred,
                                text,
                                extra={"flowmesh_stream": stream},
                            )
            except Exception:
                target_logger.debug("stdio forwarder crashed", exc_info=True)

        t = threading.Thread(target=_run, name=f"StdIOForwarder({level})", daemon=True)
        t.start()
        return t

    @staticmethod
    def _infer_level_from_line(text: str, default: int) -> int:
        """Infer a logging level from a line when possible.

        vLLM commonly prefixes lines with a level token like:
          "INFO 01-01 00:00:00 [file.py:123] ..."
        """

        s = text.lstrip()
        # Drop leading bracket / paren prefixes (e.g., `[rank0]`, `(EngineCore pid=1)`)
        for _ in range(3):
            if s.startswith("[") and "]" in s:
                close = s.find("]")
                if close > 0:
                    s = s[close + 1 :].lstrip()
                    continue
                break
            if s.startswith("(") and ")" in s:
                close = s.find(")")
                if close > 0:
                    s = s[close + 1 :].lstrip()
                    continue
                break
            break

        for name, level in logging.getLevelNamesMapping().items():
            if s.startswith(name) and (len(s) == len(name) or s[len(name)].isspace()):
                return level
            if s.startswith(f"{name}:"):
                return level
        return default


def _configure_worker_logging(log_queue: Queue | None) -> None:
    """Configure logging in worker subprocess.

    This ensures all loggers (including connectors, executors, vLLM, etc.) have
    their logs forwarded to the parent process (preferred) or written to stderr
    (fallback).

    The log level can be controlled via the WORKER_LOG_LEVEL environment variable
    (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.

    TODO: merge this with the main worker logging configuration logic.
    """
    # Get log level from environment variable, default to INFO
    log_level = os.getenv("WORKER_LOG_LEVEL", "INFO").upper()

    root_logger = logging.getLogger()

    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_queue is not None:
        qh = QueueHandler(log_queue)
        qh.setLevel(log_level)
        root_logger.addHandler(qh)
        root_logger.setLevel(log_level)
    else:
        # Fallback: stderr
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(log_level)
        # Set a formatter that includes process info
        formatter = logging.Formatter(
            "[Worker-%(process)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stderr_handler.setFormatter(formatter)  # type: ignore[name-defined]

    # Explicitly ensure common module loggers propagate to root
    for logger_name in ["worker", "connectors", "executors", "vllm"]:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(log_level)
        module_logger.propagate = True


def _executor_worker(
    executor_cls: type[Executor],
    config: WorkerConfig,
    hardware: WorkerHardware | None,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    log_queue: Queue | None,
    parent_pid: int,
) -> None:
    """Long-running subprocess that holds an executor instance and services requests.

    Protocol (tuples sent via `cmd_queue`):
      - ("run", task_payload, out_dir_str, request_id)
          -> execute `executor.run(task, Path(out_dir))` and put (request_id, payload)
          into result_queue
      - ("shutdown", request_id)
          -> call `executor.cleanup_after_run()` and exit; put (request_id, ack_payload)

    All subprocess logs are written to stderr and inherited from parent process.

    The worker keeps the executor instance alive across multiple `run` commands.

    Health check: Periodically verifies parent process is alive; exits if orphaned.
    """
    _configure_worker_logging(log_queue)
    with MPLogHandler(enabled=log_queue is not None):
        executor: Executor | None = None
        try:
            executor = executor_cls(config, hardware)
        except Exception:
            logger.warning(
                "Failed to initialize executor in worker process", exc_info=True
            )
            executor = None

        last_health_check = time.time()
        health_check_interval = 5.0  # seconds

        while True:
            # Health check: verify parent process is still alive
            now = time.time()
            if now - last_health_check >= health_check_interval:
                last_health_check = now
                if not psutil.pid_exists(parent_pid):
                    if executor is not None:
                        try:
                            executor.cleanup_after_run()
                        except Exception:
                            pass
                    if log_queue is not None:
                        try:
                            log_queue.put(None)
                        except Exception:
                            pass
                    break

            try:
                cmd = cmd_queue.get(timeout=health_check_interval)
            except Exception:
                # Timeout or error; continue to health check
                continue

            if not isinstance(cmd, (list, tuple)) or not cmd:
                logger.warning("Invalid command received in executor worker: %r", cmd)
                continue
            op = cmd[0]
            if op == "run":
                _, task, out_dir, req_id = cmd
                if executor is None:
                    payload = {
                        "ok": False,
                        "error": {
                            "type": "ExecutionError",
                            "message": "Executor failed to initialize in worker "
                            "process",
                            "traceback": "",
                            "is_execution_error": True,
                        },
                    }
                    result_queue.put((req_id, payload))
                    continue
                try:
                    result = executor.run(task, Path(out_dir))
                    payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                            "is_execution_error": isinstance(exc, ExecutionError),
                            "retryable": (
                                exc.retryable
                                if isinstance(exc, ExecutionError)
                                else False
                            ),
                        },
                    }
                result_queue.put((req_id, payload))

            elif op == "shutdown":
                _, req_id = cmd
                ack = {"ok": True}
                if executor is not None:
                    try:
                        executor.cleanup_after_run()
                    except Exception:
                        logger.warning(
                            "Exception during executor cleanup in worker process",
                            exc_info=True,
                        )
                result_queue.put((req_id, ack))
                if log_queue is not None:
                    try:
                        log_queue.put(None)
                    except Exception:
                        pass
                break

            else:
                logger.warning("Unknown command received in executor worker: %r", cmd)


class MPExecutor(Executor):
    """Executor wrapper that isolates the inner executor in a persistent subprocess.

    The subprocess holds a single instance of the inner executor and services
    multiple `run` requests. Calling `cleanup_after_run()` on this wrapper will
    shut down the subprocess (and the inner executor).

    All logs from the subprocess are written to stderr and can be captured by the
    server logging system or redirected to files.
    """

    def __init__(
        self,
        executor_cls: type[Executor],
        config: WorkerConfig,
        hardware: WorkerHardware | None = None,
        start_method: str = "spawn",
    ) -> None:
        super().__init__(config, hardware)
        self._executor_cls = executor_cls
        self._ctx = mp.get_context(start_method)
        inner_name = getattr(executor_cls, "name", executor_cls.__name__)
        self.name = f"mp({inner_name})"

        # request id counter and lock to protect subprocess operations
        self._next_req_id = 1
        self._lock = threading.Lock()
        self._shutdown = True

        # command and result queues (will be recreated on restart)
        self._cmd_q: Queue | None = None
        self._res_q: Queue | None = None
        self._log_q: Queue | None = None
        self._proc: mp.Process | None = None
        self._log_thread: threading.Thread | None = None

        # Configure logging for vLLM if applicable
        _maybe_handle_vllm_logging(executor_cls)

    def _start_process(self) -> None:
        """Start or restart the worker subprocess."""
        # Create new queues for this subprocess instance
        self._cmd_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._log_q = self._ctx.Queue()

        proc: mp.Process = self._ctx.Process(  # type: ignore
            target=_executor_worker,
            args=(
                self._executor_cls,
                self._config,
                self._hardware,
                self._cmd_q,
                self._res_q,
                self._log_q,
                os.getpid(),
            ),
        )
        proc.start()
        self._proc = proc
        self._start_log_forwarder()
        self._shutdown = False
        logger.info(
            "Started worker process (PID: %s) for %s", self._proc.pid, self.name
        )

    def _start_log_forwarder(self) -> None:
        if self._log_thread and self._log_thread.is_alive():
            return
        log_q = self._log_q
        if log_q is None:
            return

        def _loop() -> None:
            while True:
                try:
                    item = log_q.get()
                except Exception:
                    break
                if item is None:
                    break
                if isinstance(item, logging.LogRecord):
                    # Dispatch via the record's named logger so propagation
                    # reaches parent loggers that carry the real handlers.
                    logging.getLogger(item.name).handle(item)

        t = threading.Thread(target=_loop, name="MPExecutorLogForwarder", daemon=True)
        self._log_thread = t
        t.start()

    def run(self, task: ExecutorTask, out_dir: Path) -> BaseExecutorResult:
        with self._lock:
            if self._shutdown:
                logger.info("Starting worker subprocess for %s", self.name)
                self._start_process()

            cmd_q = self._cmd_q
            res_q = self._res_q
            if cmd_q is None or res_q is None:
                raise RuntimeError(f"{self.name} subprocess is not initialized")

            req_id = self._next_req_id
            self._next_req_id += 1
            try:
                cmd_q.put(("run", task, out_dir.as_posix(), req_id))
                got_id, payload = self._await_result(res_q)
                if got_id != req_id:
                    raise RuntimeError(
                        "Mismatched response ID from worker process "
                        f"(expected {req_id}, got {got_id})"
                    )
            except ExecutionError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to execute task to worker process: {exc}"
                ) from exc

            if payload is None:
                raise ExecutionError(f"{self.name} finished without returning a result")
            if payload.get("ok"):
                return payload["result"]

            error_info: dict = payload["error"]
            message = error_info.get("message", "unknown error")
            tb = error_info.get("traceback", "")
            if error_info.get("is_execution_error"):
                # Controlled failure. Keep the subprocess warm for the next task.
                raise ExecutionError(
                    message, retryable=error_info.get("retryable", False)
                )
            # An unexpected exception may have left the inner executor's engine or
            # GPU context corrupted. Restart the subprocess so the next task gets a
            # clean one.
            self._teardown_process_locked()
            raise RuntimeError(f"{self.name} failed: {message}\n{tb}")

    def _await_result(self, res_q: Queue) -> tuple[int, dict | None]:
        """Wait for a result, surfacing a dead subprocess as an error."""
        proc = self._proc
        assert proc is not None
        while True:
            try:
                return res_q.get(timeout=_RESULT_POLL_INTERVAL_SEC)
            except queue.Empty:
                if not proc.is_alive():
                    exitcode = proc.exitcode
                    self._teardown_process_locked()
                    raise ExecutionError(
                        f"{self.name} subprocess exited (code {exitcode})",
                        retryable=True,
                    ) from None

    def cleanup_after_run(self) -> None:
        """Shutdown the child process and wait for it to exit."""
        with self._lock:
            if self._shutdown:
                return  # Already cleaned up, make this idempotent

            cmd_q = self._cmd_q
            res_q = self._res_q
            proc = self._proc

            # Attempt a graceful shutdown handshake while the child is responsive;
            # _teardown_process_locked force-stops whatever remains.
            if proc and proc.is_alive() and cmd_q is not None:
                req_id = self._next_req_id
                self._next_req_id += 1
                try:
                    cmd_q.put(("shutdown", req_id), timeout=1.0)
                    logger.info("Sent shutdown command to worker process")
                    if res_q is not None:
                        res_q.get(timeout=10.0)
                        logger.debug("Received shutdown acknowledgment from worker")
                except Exception:
                    logger.warning(
                        "Failed to receive shutdown acknowledgment, proceed to force "
                        "shutdown"
                    )
                proc.join(timeout=10.0)

            self._teardown_process_locked()

    def _teardown_process_locked(self) -> None:
        """Force-stop the subprocess and log forwarder. Caller holds ``self._lock``.

        Idempotent and safe to call on an already-dead child. Clears process and
        queue references.
        """
        self._shutdown = True

        if proc := self._proc:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            logger.debug("Worker process exited with code %s", proc.exitcode)

        # A killed child can't post the sentinel that stops the log forwarder, so
        # post it here before joining the thread.
        if (log_q := self._log_q) is not None:
            try:
                log_q.put(None)
            except Exception:
                pass
        if log_thread := self._log_thread:
            try:
                log_thread.join(timeout=2.0)
            except Exception:
                pass

        self._proc = None
        self._cmd_q = None
        self._res_q = None
        self._log_q = None
        self._log_thread = None


def _maybe_handle_vllm_logging(executor_cls: type[Executor]) -> None:
    # vLLM starts its own subprocesses (engine core, workers). Those logs are
    # otherwise hard to capture via LogRecord forwarding since they are not
    # part of our MPExecutor worker process. Enable vLLM's StreamHandler-based
    # logging and rely on stdio capture inside the subprocess to forward
    # those lines to the parent.
    module_name = getattr(executor_cls, "__module__", "") or ""
    if any(
        token in module_name
        for token in (
            "worker.executors.vllm_executor",
            "worker.executors.vllm_lora_executor",
            "src.worker.executors.vllm_executor",
            "src.worker.executors.vllm_lora_executor",
        )
    ):
        if os.environ.get("VLLM_CONFIGURE_LOGGING") != "1":
            os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
        os.environ.setdefault(
            "VLLM_LOGGING_LEVEL", os.getenv("WORKER_LOG_LEVEL", "INFO").upper()
        )
        os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
