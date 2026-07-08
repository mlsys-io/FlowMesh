"""WorkerSupervisor — runs supervisor components in a child process."""

import asyncio
import logging
import os
import signal
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MPQueue
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from threading import Thread

from shared.schemas.command import CommandMessage, CommandResponse

from ..config import (
    GrpcConfig,
    IdentityConfig,
    LoggingConfig,
    RedisConfig,
    WorkerManagementConfig,
)
from ..hooks import PrincipalContext
from ..utils.concurrent import (
    MP_CTX,
    TaskReceiver,
    TaskSender,
    create_task_channel,
)

_CMD_TIMEOUT = 120.0
_NODE_ID_HANDSHAKE_TIMEOUT = 30.0
_NODE_ID_WATCH_POLL_SEC = 0.5
_REBIND_APPLY_TIMEOUT_SEC = 2.0


class WorkerSupervisor:
    """Manages the worker lifecycle on this node in a child process."""

    def __init__(
        self,
        identity: IdentityConfig,
        redis: RedisConfig,
        grpc: GrpcConfig,
        worker_management: WorkerManagementConfig,
        logging_config: LoggingConfig,
        logger: logging.Logger,
    ) -> None:
        self._identity = identity
        self._redis = redis
        self._grpc = grpc
        self._worker_management = worker_management
        self._logging_config = logging_config
        self._logger = logger
        self._process: BaseProcess | None = None
        self._cmd_sender: TaskSender[CommandMessage, CommandResponse] | None = None
        self._cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None = None
        self._node_id_queue: MPQueue[str] = MP_CTX.Queue(maxsize=8)
        self._node_id: str | None = None
        self._node_id_listeners: list[Callable[[str], None]] = []
        self._node_id_watcher: Thread | None = None
        self._node_id_stop = False

    @property
    def node_id(self) -> str:
        """Return the current node_id. Reflects re-registrations after `start()`."""
        if self._node_id is None:
            raise RuntimeError("Supervisor not started; node_id not yet assigned")
        return self._node_id

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, system_principal: PrincipalContext) -> None:
        """Spawn the supervisor child process."""
        if self._process is not None and self._process.is_alive():
            self._logger.warning("Supervisor process already running")
            return

        self._cmd_sender, self._cmd_receiver = create_task_channel()
        self._process = MP_CTX.Process(
            target=_run_supervisor,
            kwargs={
                "identity": self._identity,
                "redis_cfg": self._redis,
                "grpc_cfg": self._grpc,
                "wm_cfg": self._worker_management,
                "log_cfg": self._logging_config,
                "cmd_receiver": self._cmd_receiver,
                "node_id_queue": self._node_id_queue,
                "system_principal": system_principal,
            },
            name=f"supervisor-{self._identity.alias}",
            daemon=True,
        )
        self._process.start()
        self._cmd_sender.start()
        self._logger.info(
            "Supervisor process started (pid=%s) for node %s",
            self._process.pid,
            self._identity.alias,
        )

        try:
            node_id = await asyncio.to_thread(
                self._node_id_queue.get, True, _NODE_ID_HANDSHAKE_TIMEOUT
            )
        except QueueEmpty as exc:
            alive = self._process.is_alive()
            if alive:
                self._process.terminate()
                self._process.join(timeout=3.0)
            self._process = None
            raise RuntimeError(
                f"Supervisor child did not register a node within "
                f"{_NODE_ID_HANDSHAKE_TIMEOUT:.0f}s "
                f"(child {'still alive' if alive else 'exited'})"
            ) from exc
        self._node_id = node_id
        self._logger.info("Supervisor handshake complete: node_id=%s", node_id)

        self._node_id_stop = False
        self._node_id_watcher = Thread(
            target=self._watch_node_id, name="SupervisorNodeIdWatcher", daemon=True
        )
        self._node_id_watcher.start()

    async def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop the supervisor child process."""
        self._node_id_stop = True
        if self._node_id_watcher is not None:
            await asyncio.to_thread(self._node_id_watcher.join)
            self._node_id_watcher = None

        proc = self._process
        if (
            proc is None
            or not proc.is_alive()
            or self._cmd_sender is None
            or self._cmd_receiver is None
        ):
            return

        self._cmd_sender.stop()
        self._cmd_sender = None
        self._cmd_receiver.stop()
        self._cmd_receiver = None

        self._logger.info("Stopping supervisor process (pid=%s) ...", proc.pid)
        proc.terminate()
        proc.join(timeout=timeout)
        if proc.is_alive():
            self._logger.warning(
                "Supervisor process did not exit in %.1fs; killing", timeout
            )
            proc.kill()
            proc.join(timeout=3.0)
        self._process = None
        self._node_id = None
        self._logger.info("Supervisor process stopped")

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ------------------------------------------------------------------ #
    # Parent-side command execution
    # ------------------------------------------------------------------ #

    async def exec_cmd(
        self, cmd: CommandMessage, timeout: float = _CMD_TIMEOUT
    ) -> CommandResponse:
        """Send a command to the supervisor child via IPC queue."""
        sender = self._cmd_sender
        if sender is None:
            raise RuntimeError("Supervisor command channel not initialized")
        return await sender.send(cmd.command_id, cmd, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Node ID watching
    # ------------------------------------------------------------------ #

    def add_node_id_listener(self, listener: Callable[[str], None]) -> None:
        """Register a hook invoked with the new node_id whenever the child re-registers
        under a fresh id. Lets the parent refresh its cached copies."""
        self._node_id_listeners.append(listener)

    def _watch_node_id(self) -> None:
        """Drain the node_id queue for re-registrations after the initial handshake,
        updating the cached id and notifying listeners."""
        while not self._node_id_stop:
            try:
                new_id = self._node_id_queue.get(timeout=_NODE_ID_WATCH_POLL_SEC)
            except QueueEmpty:
                continue
            self._node_id = new_id
            self._logger.info("Supervisor node re-registered as %s", new_id)
            for listener in self._node_id_listeners:
                try:
                    listener(new_id)
                except Exception as exc:
                    self._logger.warning("node_id listener failed: %s", exc)


# ------------------------------------------------------------------ #
# Child-process entry point
# ------------------------------------------------------------------ #


def _enqueue_latest_node_id(
    queue: MPQueue[str], node_id: str, logger: logging.Logger
) -> None:
    """Put node_id on the handshake queue, dropping the stale head when it is
    full so the parent always learns the latest id."""
    try:
        queue.put_nowait(node_id)
    except QueueFull:
        try:
            queue.get_nowait()
        except QueueEmpty:
            pass
        try:
            queue.put_nowait(node_id)
        except QueueFull:
            logger.warning(
                "node_id queue full; parent will sync on the next re-register"
            )


def _run_supervisor(
    identity: IdentityConfig,
    redis_cfg: RedisConfig,
    grpc_cfg: GrpcConfig,
    wm_cfg: WorkerManagementConfig,
    log_cfg: LoggingConfig,
    cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None,
    node_id_queue: MPQueue[str],
    system_principal: PrincipalContext,
) -> None:
    """Target function executed inside the child process."""
    from pathlib import Path

    from shared._version import FLOWMESH_RELEASE_VERSION
    from shared.schemas.node import NodeInfo
    from shared.utils.time import now_iso

    from ..clients import RedisClient
    from ..registries.node import NodeRegistry
    from ..utils.logging import get_logger as _get_logger
    from .manager import WorkerManager
    from .registry import WorkerRegistry as WorkerAdapterRegistry
    from .resource_manager import ResourceManager
    from .services.command_listener import CommandListener
    from .services.grpc_server import GrpcServer
    from .services.lifecycle import Lifecycle
    from .services.relay_service import RelayService
    from .services.ssh_relay import SshRelayService
    from .services.task_listener import TaskListener

    # --- logging (child has its own logger) ---
    sv_log_file = log_cfg.file.replace("server", "supervisor")
    Path(sv_log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = _get_logger(
        name="supervisor",
        log_file=sv_log_file,
        max_bytes=log_cfg.max_bytes,
        backup_count=log_cfg.backup_count,
        level=log_cfg.level,
    )
    logger.info("Supervisor child process starting (pid=%s) ...", os.getpid())

    # --- Redis (fresh connections for this process) ---
    redis_client = RedisClient(
        control_url=redis_cfg.control_url,
        telemetry_url=redis_cfg.telemetry_url,
        logger=logger,
        acl_enabled=redis_cfg.acl_enabled,
        username=redis_cfg.username,
        password=redis_cfg.password,
        tls_ca_file=redis_cfg.tls_ca_file,
    )

    # --- GPU detection & NodeInfo construction ---
    max_gpu_count = 0
    current_gpu_count_getter = None
    try:
        rm = ResourceManager.get_instance()
        max_gpu_count = rm.total_gpu_count()
        current_gpu_count_getter = rm.available_gpu_count
    except Exception as exc:
        logger.warning("Failed to detect node GPU capacity: %s", exc)

    node_started_at = now_iso()
    node_info = NodeInfo(
        namespace=identity.namespace,
        cluster=identity.cluster,
        alias=identity.alias,
        version=FLOWMESH_RELEASE_VERSION,
        started_at=node_started_at,
        tags=identity.tags,
        last_seen=node_started_at,
        max_gpu_count=max_gpu_count,
    )

    # --- NodeRegistry (for lifecycle self-registration) ---
    node_registry = NodeRegistry(redis_client, logger)

    # --- Lifecycle: register node and get auto-assigned node_id ---
    hb_ttl_sec = max(wm_cfg.heartbeat_interval * 4, 120)

    lifecycle = Lifecycle(
        redis=redis_client.sync,
        node_registry=node_registry,
        node_info=node_info,
        role=identity.role,
        base_url=identity.base_url,
        hb_sec=wm_cfg.heartbeat_interval,
        hb_ttl_sec=hb_ttl_sec,
        logger=logger,
        system_principal=system_principal,
        current_gpu_count_getter=current_gpu_count_getter,
    )

    # Register now — must happen before other components that need node_id
    node_id = lifecycle.start()
    logger.info("Node registered as %s", node_id)

    # Hand node_id back to the parent process
    node_id_queue.put(node_id)

    # --- Supervisor components (constructed with the assigned node_id) ---
    worker_adapter_registry = WorkerAdapterRegistry()
    relay_service = RelayService(redis=redis_client.sync, logger=logger)
    ssh_relay = SshRelayService(logger=logger)
    task_listener = TaskListener(
        redis=redis_client.sync, node_id=node_id, logger=logger
    )
    worker_manager = WorkerManager(
        system_principal,
        wm_cfg.config_path,
        worker_adapter_registry,
        logger,
        capacity_change_callback=lifecycle.heartbeat_now,
    )
    command_listener = CommandListener(
        redis=redis_client.sync,
        node_id=node_id,
        worker_manager=worker_manager,
        logger=logger,
        cmd_receiver=cmd_receiver,
        ssh_relay=ssh_relay,
    )
    grpc_server = GrpcServer(
        grpc_cfg.host,
        grpc_cfg.port,
        worker_adapter_registry,
        redis=redis_client.sync,
        node_id=node_id,
        node_alias=identity.alias,
        task_listener=task_listener,
        relay_service=relay_service,
        logger=logger,
    )

    def _on_reregister(new_node_id: str) -> None:
        # Rebind the subscriptions first and wait for the reader threads to
        # switch channels, THEN re-home workers, so the dispatcher never
        # publishes to a dispatch channel nobody is listening on yet.
        task_listener.rebind(new_node_id)
        command_listener.rebind(new_node_id)
        # The rebind is reader-owned and idempotent; a timeout means the reader
        # is unresponsive, not that the switch was lost. Surface it and proceed
        # rather than stall the heartbeat thread.
        if not task_listener.wait_rebound(_REBIND_APPLY_TIMEOUT_SEC):
            logger.error(
                "Task dispatch did not rebind to %s within %ss",
                new_node_id,
                _REBIND_APPLY_TIMEOUT_SEC,
            )
        if not command_listener.wait_rebound(_REBIND_APPLY_TIMEOUT_SEC):
            logger.error(
                "Command channel did not rebind to %s within %ss",
                new_node_id,
                _REBIND_APPLY_TIMEOUT_SEC,
            )
        grpc_server.rebind_node(new_node_id)
        # Propagate the new id to the parent process (non-blocking: this runs on
        # the heartbeat thread, which must not stall on a full queue).
        _enqueue_latest_node_id(node_id_queue, new_node_id, logger)

    # --- Event loop with signal handling ---
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    async def _run() -> None:
        # Startup — lifecycle already started above (registration is synchronous)
        ssh_relay.start()
        relay_service.start()
        task_listener.start()
        await worker_manager.start()
        command_listener.start()
        await grpc_server.start()
        # Wire the re-register callback only once the reader threads are up
        lifecycle.set_reregister_callback(_on_reregister)
        logger.info("Supervisor ready for node %s", node_id)

        # Wait for termination signal
        await stop_event.wait()

        # Shutdown — reverse order
        logger.info("Supervisor shutting down ...")
        # Publish unregister event early to allow the server to handle before being
        # timed out
        lifecycle.publish_unregister()
        await grpc_server.stop()
        await command_listener.stop()
        await worker_manager.stop()
        relay_service.stop()
        await ssh_relay.stop()
        task_listener.stop()
        lifecycle.stop()
        logger.info("Supervisor stopped for node %s", node_id)

    try:
        loop.run_until_complete(_run())
    except Exception:
        logger.exception("Supervisor process crashed")
    finally:
        loop.close()
