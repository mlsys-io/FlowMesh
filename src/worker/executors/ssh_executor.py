"""SSH session executor.

Supports two modes:

**Interactive** (default): Creates an ephemeral session running sshd, emits a
TASK_UPDATE event with connection info, and blocks until the session ends (TTL,
idle timeout, or worker shutdown).

**Non-interactive** (``interactive=false``): Runs a user-provided container
image with an optional custom entrypoint/command.

A session backend (``worker.executors.ssh_session``) supplies the sandbox the
session runs in, selected by ``SSH_SESSION_BACKEND``.
"""

import logging
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared.schemas.result import SSHResult
from shared.tasks.specs.ssh import SSHSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils import new_ssh_session_id
from shared.utils.manifest import ARTIFACTS_DIR, prepare_output_dir
from worker.config import WorkerConfig
from worker.executors.ssh_session import (
    ResolvedSSHInput,
    SessionRequest,
    SSHConfig,
    SSHOutputConfig,
    SSHSession,
    SSHSessionBackend,
    select_backend_cls,
)
from worker.executors.ssh_session.inputs import resolve_inputs
from worker.executors.utils.checkpoints import maybe_upload_artifacts

from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    TaskCancelledError,
)

logger = logging.getLogger(__name__)

_SESSION_READY_TIMEOUT_SEC = 30.0

__all__ = ["ResolvedSSHInput", "SSHConfig", "SSHExecutor", "SSHOutputConfig"]


class SSHExecutor(Executor):
    """Executor for SSH tasks (interactive sessions and non-interactive jobs)."""

    name = "ssh"
    supported_task_types = frozenset({TaskType.SSH})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self._config
        self._worker_name = config.container_name or config.alias or None
        self._cancel_event = threading.Event()
        self._finish_event = threading.Event()
        self._current_session: SSHSession | None = None
        self._backend = self._make_backend(config)

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return select_backend_cls(config) is not None

    @property
    def worker_name(self) -> str:
        if name := self._worker_name:
            return name
        name = (
            lifecycle.worker_id
            if (lifecycle := self._lifecycle)
            else uuid.uuid4().hex[:8]
        )
        self._worker_name = name
        return name

    @property
    def backend(self) -> SSHSessionBackend:
        return self._backend

    def _make_backend(self, config: WorkerConfig) -> SSHSessionBackend:
        backend_cls = select_backend_cls(config)
        if backend_cls is None:
            raise ExecutionError(
                "No SSH session backend is available on this worker "
                f"(SSH_SESSION_BACKEND={config.ssh_session_backend!r})"
            )
        logger.info("SSH sessions use the %s backend", backend_cls.name)
        return backend_cls(config, self._hardware)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        self._backend.prepare()

    def teardown(self) -> None:
        """Stop all sessions owned by this worker."""
        self._backend.teardown(self.worker_name)

    # ------------------------------------------------------------------ #
    # Main execution
    # ------------------------------------------------------------------ #

    def run(self, task: ExecutorTask, out_dir: Path) -> SSHResult:
        spec = self.require_spec(task, SSHSpecStrict)
        cfg = SSHConfig.from_spec(spec, self._config, self._hardware)
        access_mode = cfg.access_mode
        interactive = cfg.interactive

        if interactive and access_mode not in ("direct", "proxy", "forward"):
            raise ExecutionError(f"accessMode '{access_mode}' is not supported")

        self.prepare()
        session_id = new_ssh_session_id()

        prepare_output_dir(out_dir)  # Ensure output dir exists before mounting
        resolved_inputs = resolve_inputs(task, cfg, self._config.results_dir)
        request = SessionRequest(
            task_id=task.task_id,
            session_id=session_id,
            worker_name=self.worker_name,
            cfg=cfg,
            out_dir=out_dir,
            resolved_inputs=resolved_inputs,
        )

        session_kind = "SSH session" if interactive else "non-interactive"
        if interactive:
            logger.info(
                "Starting %s (task=%s session=%s mode=%s ttl=%ds idle=%ds)",
                session_kind,
                task.task_id,
                session_id,
                access_mode,
                cfg.ttl_sec,
                cfg.idle_sec,
            )
        else:
            logger.info(
                "Starting %s session (task=%s session=%s ttl=%ds cmd=%s)",
                session_kind,
                task.task_id,
                session_id,
                cfg.ttl_sec,
                cfg.command,
            )

        try:
            session = self._backend.start_session(request)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"Failed to start {session_kind}: {exc}") from exc

        self._current_session = session
        log_thread: threading.Thread | None = None
        if not interactive:
            log_thread = threading.Thread(
                target=session.drain_logs,
                daemon=True,
                name=f"flowmesh-session-logs-{task.task_id[:8]}",
            )
            log_thread.start()
        exit_code = 0
        try:
            session_info = (
                self._wait_session_ready(session, session_id, task, cfg)
                if interactive
                else {}
            )
            exit_code = self._wait_for_session(session, cfg)
            result = SSHResult(session_id=session_id, exit_code=exit_code)
            if interactive:
                for key, value in session_info.items():
                    setattr(result, key, value)
            else:
                # Keep as fallback — captures any output the streaming thread missed.
                session.save_logs(out_dir)
                if cfg.command is not None:
                    result.command = cfg.command
                if cfg.entrypoint is not None:
                    result.entrypoint = cfg.entrypoint
            session.collect_output(out_dir / ARTIFACTS_DIR)
            maybe_upload_artifacts(task, out_dir, logger=logger, skip_errors=True)
        finally:
            if log_thread is not None:
                # Wait for the thread to drain remaining output before tearing down
                # the session.
                log_thread.join(timeout=30.0)
            self.withdraw_endpoint(session_id)
            self._current_session = None
            self._cancel_event.clear()
            self._finish_event.clear()
            session.stop(cfg.stop_timeout_sec)
            session.cleanup()

        if not (interactive or exit_code == 0):
            raise ExecutionError(
                f"{session_kind.capitalize()} session exited with code {exit_code}"
            )

        return result

    def cancel(self, task_id: str) -> None:
        self._cancel_event.set()
        self._stop_current_session("cancellation")

    def stop(self, task_id: str) -> None:
        self._finish_event.set()
        self._stop_current_session("graceful stop")

    def _stop_current_session(self, reason: str) -> None:
        session = self._current_session
        if session is None:
            return
        try:
            session.stop(1)
        except Exception:
            logger.debug("Failed to stop SSH session during %s", reason, exc_info=True)

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def _wait_session_ready(
        self, session: SSHSession, session_id: str, task: ExecutorTask, cfg: SSHConfig
    ) -> dict[str, Any]:
        access_mode = cfg.access_mode
        expires_at = self._iso_offset(cfg.ttl_sec)
        host_port = session.wait_ready(_SESSION_READY_TIMEOUT_SEC)
        self.publish_endpoint(session_id, host_port)
        host_name = self._backend.session_host()
        ssh_info: dict[str, Any] = {
            "session_id": session_id,
            "mode": access_mode,
            "username": session.login_user(),
            "expires_at": expires_at,
            "host": host_name,
            "port": host_port,
        }
        if access_mode in ("proxy", "forward"):
            relay_host = self._backend.relay_host()
            if access_mode == "forward":
                # Forward-mode sessions need separate direct connection info
                ssh_info["directHost"] = host_name
                ssh_info["directPort"] = host_port
            ssh_info["_relay_target"] = {
                "host": relay_host,
                "port": host_port,
            }
            logger.info(
                "SSH %s session ready: host=%s port=%s relay=%s (task=%s)",
                access_mode,
                host_name,
                host_port,
                relay_host,
                task.task_id,
            )
        else:
            logger.info(
                "SSH session ready: host=%s port=%s (task=%s)",
                host_name,
                host_port,
                task.task_id,
            )
        self.emit_update(task.task_id, {"ssh": ssh_info})
        return {
            "expires_at": expires_at,
            "host": None if host_port is None else host_name,
            "port": host_port,
        }

    def _wait_for_session(self, session: SSHSession, cfg: SSHConfig) -> int:
        """Block until the session exits or TTL/idle timeout fires.

        Returns the session exit code. The idle clock starts when the session
        does, so a session nobody ever connects to is reaped too.
        """
        deadline = time.time() + cfg.ttl_sec
        idle_enabled = cfg.interactive and cfg.idle_sec > 0
        last_active = time.time()
        idle_unobservable_logged = False
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("SSH session cancelled")
            if self._finish_event.is_set() or session.finish_requested():
                logger.info("SSH session finish requested; stopping session")
                session.stop(1)
                return 0
            self._enforce_output_limit(session, cfg.output)
            try:
                exit_code = session.poll()
            except Exception as exc:
                logger.debug("Session poll error (may have exited): %s", exc)
                if self._finish_event.is_set():
                    return 0
                break
            if exit_code is not None:
                return exit_code
            if idle_enabled:
                connections = session.established_connections()
                if connections is None:
                    if not idle_unobservable_logged:
                        logger.warning(
                            "SSH idle timeout cannot be enforced: this session's "
                            "connection state is not observable"
                        )
                        idle_unobservable_logged = True
                    last_active = time.time()
                elif connections > 0:
                    last_active = time.time()
                elif time.time() - last_active >= cfg.idle_sec:
                    logger.info(
                        "SSH session idle for %ds; stopping session", cfg.idle_sec
                    )
                    session.stop(1)
                    return 0
            time.sleep(cfg.poll_interval_sec)

        logger.info("SSH session TTL reached; stopping session")
        return 0

    def _enforce_output_limit(
        self, session: SSHSession, output_cfg: SSHOutputConfig | None
    ) -> None:
        if output_cfg is None or output_cfg.max_bytes is None:
            return
        max_bytes = output_cfg.max_bytes
        if max_bytes < 0:
            logger.warning(
                "Invalid maxBytes %d in SSH output config; ignoring limit", max_bytes
            )
            return

        current_size = session.output_size_bytes()
        if current_size is None or current_size <= max_bytes:
            return

        logger.warning(
            "SSH output exceeded maxBytes (%d > %d)", current_size, max_bytes
        )
        session.stop(1)
        raise ExecutionError(
            f"SSH sshOutput exceeded maxBytes ({current_size} > {max_bytes})"
        )

    @staticmethod
    def _iso_offset(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
