"""vLLM OpenAI-compatible serving executor.

Starts a persistent vLLM API server for a single model, emits a TASK_UPDATE
with the endpoint details, and blocks until the TTL expires or a stop command
arrives.
"""

import collections
import importlib.metadata
import logging
import os
import secrets
import signal
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs.serve import ServeSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import parse_float_env
from worker.config import WorkerConfig

from .base_executor import ExecutionError, Executor, ExecutorTask, TaskCancelledError

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SEC = 3600.0
_MAX_TTL_SEC = 86400.0
_DEFAULT_PORT = 8000
_HEALTH_POLL_INTERVAL_SEC = 2.0
# 600s default: cold-start includes model download, engine init, and CUDA graph capture
_DEFAULT_READINESS_TIMEOUT_SEC = 600.0
_POLL_INTERVAL_SEC = 5.0
_STOP_TIMEOUT_SEC = 15.0
_TAIL_MAX_LINES = 200
_TAIL_SNIPPET_BYTES = 4096


def _drain_to_log(
    proc: "subprocess.Popen[str]",
    tail: "collections.deque[str]",
    eof_event: threading.Event,
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        logger.info("[vllm] %s", line)
        tail.append(line)
    eof_event.set()


def _tail_snippet(tail: "collections.deque[str]") -> str:
    text = "\n".join(tail)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) > _TAIL_SNIPPET_BYTES:
        text = "...\n" + raw[-_TAIL_SNIPPET_BYTES:].decode("utf-8", errors="replace")
    return text


class ServeResult(BaseExecutorResult):
    model: str
    port: int
    api_key: str


class VLLMServeExecutor(Executor):
    name = "vllm_serve"
    supported_task_types = frozenset({TaskType.SERVE})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._proc: subprocess.Popen | None = None  # type: ignore[type-arg]

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _vllm_plugins_excluding_omni() -> str:
        names: list[str] = []
        for ep in importlib.metadata.entry_points(group="vllm.general_plugins"):
            module = ep.value.split(":", 1)[0].strip()
            if module == "vllm_omni" or module.startswith("vllm_omni."):
                continue
            names.append(ep.name)
        return ",".join(names)

    def run(self, task: ExecutorTask, out_dir: Path) -> ServeResult:
        spec = self.require_spec(task, ServeSpecStrict)

        ttl_sec = min(
            spec.ttlSeconds
            or parse_float_env("SERVE_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC),
            parse_float_env("SERVE_MAX_TTL_SEC", _MAX_TTL_SEC),
        )
        readiness_timeout = (
            spec.readinessTimeoutSeconds or _DEFAULT_READINESS_TIMEOUT_SEC
        )
        port = spec.port or _DEFAULT_PORT
        access_mode = spec.accessMode or "forward"
        api_key = secrets.token_hex(32)

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            spec.model,
            "--port",
            str(port),
            "--api-key",
            api_key,
        ]
        for k, v in (spec.vllmArgs or {}).items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])

        env = dict(os.environ)
        env["VLLM_PLUGINS"] = self._vllm_plugins_excluding_omni()
        env.setdefault("VLLM_CONFIGURE_LOGGING", "0")
        env["PYTHONUNBUFFERED"] = "1"

        logger.info(
            "Starting vLLM server for model %s on port %d "
            "(task=%s ttl=%.0fs readiness_timeout=%.0fs)",
            spec.model,
            port,
            task.task_id,
            ttl_sec,
            readiness_timeout,
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        if self._stop_event.is_set():
            raise TaskCancelledError(
                f"Serve task {task.task_id} stopped before vLLM launch"
            )

        tail: collections.deque[str] = collections.deque(maxlen=_TAIL_MAX_LINES)
        try:
            proc = subprocess.Popen(  # nosec B603 - argv list, no shell=True, absolute path via sys.executable
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise ExecutionError(f"Failed to start vLLM server: {exc}") from exc

        eof_event = threading.Event()
        drain_thread = threading.Thread(
            target=_drain_to_log, args=(proc, tail, eof_event), daemon=True
        )
        drain_thread.start()

        self._proc = proc
        try:
            self._poll_health(
                proc, port, task.task_id, readiness_timeout, tail, eof_event
            )
            update_payload: dict[str, Any] = {
                "serve": {
                    "mode": access_mode,
                    "_relay_target": {"host": "127.0.0.1", "port": port},
                    "host": "127.0.0.1",
                    "port": port,
                    "api_key": api_key,
                    "model": spec.model,
                }
            }
            self.emit_update(task.task_id, update_payload)
            logger.info("vLLM server ready on port %d (task=%s)", port, task.task_id)
            self._wait_for_serve(proc, ttl_sec)
        finally:
            self._proc = None
            self._cancel_event.clear()
            self._stop_event.clear()
            self._terminate_process_group(proc)
            drain_thread.join(timeout=5.0)

        return ServeResult(model=spec.model, port=port, api_key=api_key)

    def _poll_health(
        self,
        proc: "subprocess.Popen[str]",
        port: int,
        task_id: str,
        timeout_sec: float,
        tail: "collections.deque[str]",
        eof_event: threading.Event | None = None,
    ) -> None:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("Serve task cancelled during health poll")
            if self._stop_event.is_set():
                raise TaskCancelledError("Serve task stopped during health poll")
            if proc.poll() is not None:
                snippet = _tail_snippet(tail)
                raise ExecutionError(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})"
                    + (f"\n--- last vLLM output ---\n{snippet}" if snippet else "")
                )
            if eof_event is not None and eof_event.is_set():
                # Stdout pipe closed: the whole vLLM process tree exited.
                # proc.poll() may lag slightly behind pipe close; wait briefly.
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                snippet = _tail_snippet(tail)
                raise ExecutionError(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})"
                    + (f"\n--- last vLLM output ---\n{snippet}" if snippet else "")
                )
            try:
                resp = requests.get(url, timeout=2.0)  # nosec B113 - explicit timeout
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL_SEC)
        snippet = _tail_snippet(tail)
        raise ExecutionError(
            f"vLLM server did not become ready within {timeout_sec:.0f}s "
            f"(task={task_id})"
            + (f"\n--- last vLLM output ---\n{snippet}" if snippet else "")
        )

    def _wait_for_serve(self, proc: "subprocess.Popen[str]", ttl_sec: float) -> None:
        deadline = time.time() + ttl_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("Serve task cancelled")
            if self._stop_event.is_set():
                logger.info("Serve task stop requested; terminating vLLM server")
                return
            if proc.poll() is not None:
                raise ExecutionError(
                    f"vLLM server process exited unexpectedly (code={proc.returncode})"
                )
            time.sleep(_POLL_INTERVAL_SEC)
        logger.info("Serve task TTL reached; terminating vLLM server")

    def _terminate_process_group(self, proc: "subprocess.Popen[str]") -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, ChildProcessError, OSError):
                pass
            try:
                proc.wait(timeout=_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, ChildProcessError, OSError):
                    pass
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            pass

    def cancel(self, task_id: str) -> None:
        self._cancel_event.set()
        proc = self._proc
        if proc is not None:
            self._terminate_process_group(proc)

    def stop(self, task_id: str) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc is not None:
            self._terminate_process_group(proc)
