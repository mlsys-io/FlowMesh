"""vLLM OpenAI-compatible serving executor.

Starts a persistent vLLM API server for a single model, emits a TASK_UPDATE
with the endpoint details, and blocks until the TTL expires or a stop command
arrives.
"""

import collections
import logging
import os
import secrets
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any, NoReturn

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
_HEALTH_POLL_INTERVAL_SEC = 2.0
# 600s default: cold-start includes model download, engine init, and CUDA graph capture
_DEFAULT_READINESS_TIMEOUT_SEC = 600.0
_POLL_INTERVAL_SEC = 5.0
_STOP_TIMEOUT_SEC = 15.0
_TAIL_MAX_LINES = 200
_TAIL_SNIPPET_BYTES = 4096


def _drain_to_log(
    proc: subprocess.Popen[str],
    tail: collections.deque[str],
    eof_event: threading.Event,
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        logger.info("[vllm] %s", line)
        tail.append(line)
    eof_event.set()


def _tail_snippet(tail: collections.deque[str]) -> str:
    text = "\n".join(tail)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) > _TAIL_SNIPPET_BYTES:
        text = "...\n" + raw[-_TAIL_SNIPPET_BYTES:].decode("utf-8", errors="replace")
    return text


def _raise_with_tail(message: str, tail: collections.deque[str]) -> NoReturn:
    snippet = _tail_snippet(tail)
    raise ExecutionError(
        message + (f"\n--- last vLLM output ---\n{snippet}" if snippet else "")
    )


def _resolve_port(requested: int | None, bind_host: str) -> int:
    """Resolve the port vLLM binds on ``bind_host``.

    When ``requested`` is ``None`` a free ephemeral port is selected; hardcoding
    a default (e.g. 8000) collides with co-located services such as the FlowMesh
    server on a host-networked node. When ``requested`` is set but unavailable,
    raise so the caller reports a clear error instead of a raw vLLM bind failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, requested or 0))
        except OSError as exc:
            raise ExecutionError(
                f"serve port {requested} is unavailable on the worker "
                f"({bind_host}): {exc}. Choose a different spec.port, or omit it "
                "to auto-select a free port."
            ) from exc
        return probe.getsockname()[1]


class ServeResult(BaseExecutorResult):
    model: str
    port: int


class VLLMServeExecutor(Executor):
    name = "vllm_serve"
    supported_task_types = frozenset({TaskType.SERVE})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._proc: subprocess.Popen[str] | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except Exception:
            return False

    def run(self, task: ExecutorTask, out_dir: Path) -> ServeResult:
        spec = self.require_spec(task, ServeSpecStrict)

        model_id = spec.model_name
        if model_id is None:
            raise ExecutionError("Serve spec is missing model.source.identifier")

        ttl_sec = min(
            spec.ttlSeconds
            or parse_float_env("SERVE_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC),
            parse_float_env("SERVE_MAX_TTL_SEC", _MAX_TTL_SEC),
        )
        readiness_timeout = (
            spec.readinessTimeoutSeconds or _DEFAULT_READINESS_TIMEOUT_SEC
        )
        access_mode = spec.accessMode or "forward"
        api_key = spec.apiKey or secrets.token_hex(32)

        bind_host = (
            "0.0.0.0" if access_mode == "direct" else "127.0.0.1"
        )  # nosec B104 - direct mode is an explicit opt-in to a client-reachable endpoint
        port = _resolve_port(spec.port, bind_host)

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            bind_host,
            "--port",
            str(port),
            "--api-key",
            api_key,
        ]
        if revision := spec.model_revision:
            cmd.extend(["--revision", revision])

        vllm_kwargs = spec.model.vllm if spec.model is not None else None
        rendered_flags: set[str] = set()
        for k, v in (vllm_kwargs or {}).items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
                    rendered_flags.add(flag)
            else:
                cmd.extend([flag, str(v)])
                rendered_flags.add(flag)

        if spec.model_trust_remote_code and "--trust-remote-code" not in rendered_flags:
            cmd.append("--trust-remote-code")

        env = dict(os.environ)
        env.setdefault("VLLM_CONFIGURE_LOGGING", "0")
        env["PYTHONUNBUFFERED"] = "1"

        logger.info(
            "Starting vLLM server for model %s on port %d "
            "(task=%s ttl=%.0fs readiness_timeout=%.0fs)",
            model_id,
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
            advertised_host = (
                socket.getfqdn() if access_mode == "direct" else "127.0.0.1"
            )
            update_payload: dict[str, Any] = {
                "serve": {
                    "mode": access_mode,
                    "_relay_target": {"host": "127.0.0.1", "port": port},
                    "host": advertised_host,
                    "port": port,
                    "api_key": api_key,
                    "model": model_id,
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

        return ServeResult(model=model_id, port=port)

    def _poll_health(
        self,
        proc: subprocess.Popen[str],
        port: int,
        task_id: str,
        timeout_sec: float,
        tail: collections.deque[str],
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
                _raise_with_tail(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})",
                    tail,
                )
            if eof_event is not None and eof_event.is_set():
                # Stdout pipe closed: the whole vLLM process tree exited.
                # proc.poll() may lag slightly behind pipe close; wait briefly.
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                _raise_with_tail(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})",
                    tail,
                )
            try:
                resp = requests.get(url, timeout=2.0)  # nosec B113 - explicit timeout
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL_SEC)
        _raise_with_tail(
            f"vLLM server did not become ready within {timeout_sec:.0f}s "
            f"(task={task_id})",
            tail,
        )

    def _wait_for_serve(self, proc: subprocess.Popen[str], ttl_sec: float) -> None:
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

    def _terminate_process_group(self, proc: subprocess.Popen[str]) -> None:
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
