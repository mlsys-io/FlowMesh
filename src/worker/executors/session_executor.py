"""Shared lifecycle for executors that run a task inside a session sandbox.

A session backend (``worker.executors.ssh_session``) supplies the sandbox the
session runs in, selected by ``SSH_SESSION_BACKEND``. ``SessionExecutor`` brings
a session up, waits for it to end, collects its output, and tears it down;
subclasses turn the resolved config and the way the session ended into their
own result.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from shared.tasks.specs.ssh import SSHSpecStrict
from shared.tasks.worker_message import GpuInfo
from shared.utils import new_ssh_session_id
from shared.utils.manifest import ARTIFACTS_DIR, prepare_output_dir
from worker.config import WorkerConfig
from worker.executors.ssh_session import (
    SessionRequest,
    SSHConfig,
    SSHOutputConfig,
    SSHSession,
    SSHSessionBackend,
    select_backend_cls,
)
from worker.executors.ssh_session.inputs import resolve_inputs
from worker.executors.utils.checkpoints import maybe_upload_artifacts
from worker.gpu_availability import DeviceAvailability
from worker.result_delivery import rewrite_artifact_inputs

from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    TaskCancelledError,
)

logger = logging.getLogger(__name__)

_SESSION_READY_TIMEOUT_SEC = 30.0

type SessionEndReason = Literal["exited", "ttl", "idle", "finished", "lost"]


@dataclass(frozen=True)
class SessionEnd:
    """How a session ended.

    ``exited``: the session's process exited with ``exit_code``. ``ttl`` / ``idle``:
    the worker stopped it at its deadline or after it sat idle. ``finished``: a
    finish was requested (in-session helper or a graceful stop). ``lost``: the
    session could no longer be observed. Every reason but ``exited`` has exit
    code 0.
    """

    reason: SessionEndReason
    exit_code: int = 0


@dataclass(frozen=True)
class SessionOutcome:
    session_id: str
    end: SessionEnd
    ready_info: dict[str, Any] = field(default_factory=dict)


def _available_uuids(
    reported: dict[str, DeviceAvailability], devices: list[GpuInfo]
) -> frozenset[str] | None:
    """The devices a session may be given.

    ``None`` when the worker took no reading at all, which leaves the caller's
    device set untouched. A device the reading did not cover is "no opinion" rather
    than "held", so it stays in the set -- withholding it would strand sessions on a
    worker whose probe only ever covers some of its cards.
    """
    if not reported:
        return None
    return frozenset(
        device.uuid
        for device in devices
        if (seen := reported.get(device.uuid)) is None or seen.available
    )


class SessionExecutor(Executor):
    """Base for executors whose tasks run as a session on a session backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self._config
        self._owner = config.container_name or config.alias
        self._cancel_event = threading.Event()
        self._finish_event = threading.Event()
        self._current_session: SSHSession | None = None
        self._backend = self._make_backend(config)

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return select_backend_cls(config) is not None

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
        logger.info("Sessions use the %s backend", backend_cls.name)
        return backend_cls(config, self._hardware)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        self._backend.prepare()

    def teardown(self) -> None:
        """Stop all sessions owned by this worker."""
        self._backend.teardown(self._owner)

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
            logger.debug("Failed to stop session during %s", reason, exc_info=True)

    # ------------------------------------------------------------------ #
    # Running a session
    # ------------------------------------------------------------------ #

    def _config_for(self, spec: SSHSpecStrict) -> SSHConfig:
        # A session holds its devices for as long as it lives, so it must not be
        # handed one another tenant is already on.
        return SSHConfig.from_spec(
            spec,
            self._config,
            self._hardware,
            _available_uuids(
                self._lifecycle.live_gpu_availability() if self._lifecycle else {},
                self._hardware.gpu.devices if self._hardware else [],
            ),
        )

    def _run_session(
        self, task: ExecutorTask, out_dir: Path, cfg: SSHConfig
    ) -> SessionOutcome:
        """Bring a session up, wait for it to end, collect its output, tear it down."""
        access_mode = cfg.access_mode
        interactive = cfg.interactive

        if interactive and access_mode not in ("direct", "proxy", "forward"):
            raise ExecutionError(f"accessMode '{access_mode}' is not supported")

        self.prepare()
        session_id = new_ssh_session_id()

        prepare_output_dir(out_dir)  # Ensure output dir exists before mounting
        resolved_inputs = resolve_inputs(task, cfg, self._config.results_dir)
        replacements = {}
        for entry in task.artifact_inputs.get(task.task_id, []):
            local = (
                (self._config.results_dir / entry.task_id / "artifacts" / entry.path)
                .resolve()
                .as_posix()
            )
            replacements[local] = (
                f"/mnt/flowmesh/references/{entry.task_id}/artifacts/{entry.path}"
            )
        cfg.extra_env = rewrite_artifact_inputs(cfg.extra_env, replacements)
        cfg.command = rewrite_artifact_inputs(cfg.command, replacements)
        cfg.entrypoint = rewrite_artifact_inputs(cfg.entrypoint, replacements)
        request = SessionRequest(
            task_id=task.task_id,
            session_id=session_id,
            owner=self._owner,
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
        try:
            ready_info = (
                self._wait_session_ready(session, session_id, task, cfg)
                if interactive
                else {}
            )
            end = self._wait_for_session(session, cfg)
            if not interactive:
                # Keep as fallback — captures any output the streaming thread missed.
                session.save_logs(out_dir)
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

        return SessionOutcome(session_id=session_id, end=end, ready_info=ready_info)

    def _wait_session_ready(
        self, session: SSHSession, session_id: str, task: ExecutorTask, cfg: SSHConfig
    ) -> dict[str, Any]:
        access_mode = cfg.access_mode
        expires_at = self._iso_offset(cfg.ttl_sec)
        host_port = session.wait_ready(_SESSION_READY_TIMEOUT_SEC)
        self.publish_endpoint(session_id, host_port)
        host_name = self._backend.session_address(access_mode)
        ssh_info: dict[str, Any] = {
            "session_id": session_id,
            "mode": access_mode,
            "username": session.login_user(),
            "expires_at": expires_at,
            "host": host_name,
            "port": host_port,
        }
        if access_mode in ("proxy", "forward"):
            # A relayed session's own address, reported for every relayed mode
            # so a client on the worker's host can still reach it. `host` and
            # `port` may be rewritten to the server's route; these are not.
            ssh_info["directHost"] = host_name
            ssh_info["directPort"] = host_port
            ssh_info["directScope"] = self._backend.session_scope(access_mode)
            ssh_info["workerId"] = task.assigned_worker
            logger.info(
                "SSH %s session ready: host=%s port=%s (task=%s)",
                access_mode,
                host_name,
                host_port,
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

    def _wait_for_session(self, session: SSHSession, cfg: SSHConfig) -> SessionEnd:
        """Block until the session exits or its TTL/idle timeout fires.

        The idle clock starts when the session does, so a session nobody ever
        connects to is reaped too.
        """
        deadline = time.time() + cfg.ttl_sec
        idle_enabled = cfg.interactive and cfg.idle_sec > 0
        last_active = time.time()
        idle_unobservable_logged = False
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("Session cancelled")
            if cfg.honor_finish_request and (
                self._finish_event.is_set() or session.finish_requested()
            ):
                logger.info("Session finish requested; stopping session")
                session.stop(1)
                return SessionEnd("finished")
            self._enforce_output_limit(session, cfg.output)
            try:
                exit_code = session.poll()
            except Exception as exc:
                logger.debug("Session poll error (may have exited): %s", exc)
                if self._finish_event.is_set():
                    return SessionEnd("finished")
                return SessionEnd("lost")
            if exit_code is not None:
                return SessionEnd("exited", exit_code)
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
                    return SessionEnd("idle")
            time.sleep(cfg.poll_interval_sec)

        logger.info("Session TTL reached; stopping session")
        session.stop(cfg.stop_timeout_sec)
        return SessionEnd("ttl")

    def _enforce_output_limit(
        self, session: SSHSession, output_cfg: SSHOutputConfig | None
    ) -> None:
        if output_cfg is None or output_cfg.max_bytes is None:
            return
        max_bytes = output_cfg.max_bytes
        if max_bytes < 0:
            logger.warning(
                "Invalid maxBytes %d in session output config; ignoring limit",
                max_bytes,
            )
            return

        current_size = session.output_size_bytes()
        if current_size is None or current_size <= max_bytes:
            return

        logger.warning(
            "Session output exceeded maxBytes (%d > %d)", current_size, max_bytes
        )
        session.stop(1)
        raise ExecutionError(
            f"Session output exceeded maxBytes ({current_size} > {max_bytes})"
        )

    @staticmethod
    def _iso_offset(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
