import asyncio
import contextlib
import logging
import queue
import secrets
from collections.abc import Callable, Iterable
from concurrent import futures
from threading import Event, Lock, Thread

from pydantic import ValidationError
from redis.client import PubSub

from shared.schemas.command import (
    CommandMessage,
    CommandResponse,
    CommandType,
)

from ...clients.redis import (
    NODE_RESPONSE_CHANNEL,
    SyncRedisClient,
    node_cmd_channel,
    parse_pubsub_message,
)
from ...utils.concurrent import Sentinel, TaskReceiver
from ..adapters.docker import DockerWorkerConfig
from ..manager import WorkerInitConfig, WorkerManager
from .ssh_relay import SshRelayService

type ResponseHandler = Callable[[CommandResponse], None]

_MAX_INFLIGHT_CMDS = 32
_STOP_DRAIN_TIMEOUT = 10.0
_START_WORKER_TIMEOUT = 600.0
_STOP_WORKER_TIMEOUT = 60.0
_CREATE_WORKER_TIMEOUT = 600.0
_DESTROY_WORKER_TIMEOUT = 60.0
_DESTROY_WORKERS_TIMEOUT = 120.0
_POLL_TIMEOUT_SEC = 0.25


def _cmd_receiver_loop(
    receiver: TaskReceiver[CommandMessage, CommandResponse],
    q: queue.Queue[tuple[CommandMessage, ResponseHandler]],
) -> None:
    for cmd_id, cmd in receiver.task_stream():

        def make_handler(cmd_id: str) -> ResponseHandler:
            def send_response(resp: CommandResponse) -> None:
                receiver.send_result(cmd_id, resp)

            return send_response

        q.put((cmd, make_handler(cmd_id)))


class _CommandStream:
    _SENTINEL = Sentinel()

    def __init__(
        self,
        node_id: str,
        redis: SyncRedisClient,
        cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None,
        logger: logging.Logger,
    ):
        # node_id is mutated only by the pubsub reader thread once running.
        self.node_id = node_id
        self.redis = redis
        self.cmd_receiver = cmd_receiver
        self.logger = logger
        self._cmd_queue: (
            queue.Queue[tuple[CommandMessage, ResponseHandler] | Sentinel] | None
        ) = None
        self._pubsub_running = False

        self._rebind_lock = Lock()
        self._pending_node_id: str | None = None
        self._rebind_applied = Event()

    def close(self) -> None:
        if self._cmd_queue is not None:
            self._cmd_queue.put(self._SENTINEL)
            self._cmd_queue = None

    def rebind(self, node_id: str) -> None:
        """Request moving the command subscription to a new node id.

        Records the target under a lock; the reader thread applies the actual
        ``subscribe``/``unsubscribe`` between polls (mutating redis-py ``PubSub`` is not
        thread-safe). ``wait_rebound`` blocks until the switch has taken effect.
        """
        if node_id == self.node_id:
            self._rebind_applied.set()
            return
        self._rebind_applied.clear()
        with self._rebind_lock:
            self._pending_node_id = node_id

    def wait_rebound(self, timeout: float) -> bool:
        return self._rebind_applied.wait(timeout)

    def _apply_pending_rebind(self, pubsub: PubSub, current_id: str) -> str:
        with self._rebind_lock:
            pending = self._pending_node_id
            self._pending_node_id = None
        if pending is None or pending == current_id:
            return current_id
        pubsub.subscribe(node_cmd_channel(pending))
        pubsub.unsubscribe(node_cmd_channel(current_id))
        self.node_id = pending
        self._rebind_applied.set()
        self.logger.info(
            "Command stream rebound from node %s to %s", current_id, pending
        )
        return pending

    def _run_pubsub(
        self,
        pubsub: PubSub,
        cmd_queue: queue.Queue[tuple[CommandMessage, ResponseHandler] | Sentinel],
    ) -> None:
        def send_response(resp: CommandResponse) -> None:
            self.redis.publish_control(NODE_RESPONSE_CHANNEL, resp.model_dump_json())

        current_id = self.node_id
        try:
            while self._pubsub_running:
                current_id = self._apply_pending_rebind(pubsub, current_id)
                msg = pubsub.get_message(timeout=_POLL_TIMEOUT_SEC)
                data = parse_pubsub_message(msg)
                if data is None:
                    continue
                try:
                    cmd = CommandMessage.model_validate(data)
                except ValidationError as e:
                    self.logger.error("Invalid command message: %s", e)
                    continue
                cmd_queue.put((cmd, send_response))
        except (ConnectionError, OSError):
            return
        except Exception as exc:
            if self._pubsub_running:
                self.logger.exception("Command pubsub loop error: %s", exc)

    def iter_stream(self) -> Iterable[tuple[CommandMessage, ResponseHandler]]:
        if self._cmd_queue is not None:
            raise RuntimeError("Command stream already in use")
        self._cmd_queue = queue.Queue()
        cmd_queue = self._cmd_queue

        if self.cmd_receiver is not None:
            task_thread = Thread(
                target=_cmd_receiver_loop,
                args=(self.cmd_receiver, cmd_queue),
                name="CommandReceiverThread",
                daemon=True,
            )
            task_thread.start()

        pubsub = self.redis.subscribe_control(node_cmd_channel(self.node_id))
        self._pubsub_running = True
        pubsub_thread = Thread(
            target=self._run_pubsub,
            args=(pubsub, cmd_queue),
            name="CommandPubSubThread",
            daemon=True,
        )
        pubsub_thread.start()

        try:
            while True:
                item = cmd_queue.get()
                if isinstance(item, Sentinel):
                    assert item is self._SENTINEL
                    break
                yield item
        finally:
            self._pubsub_running = False
            pubsub_thread.join()
            pubsub.close()


class CommandListener:
    """Routes commands to async handlers running concurrently on the
    supervisor's event loop, with per-worker serialization for ops that
    target a named worker.
    """

    def __init__(
        self,
        redis: SyncRedisClient,
        node_id: str,
        worker_manager: WorkerManager,
        logger: logging.Logger,
        cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None = None,
        ssh_relay: SshRelayService | None = None,
        max_inflight: int = _MAX_INFLIGHT_CMDS,
    ) -> None:
        self.logger = logger
        self._redis = redis
        self._node_id = node_id
        self._wm = worker_manager
        self._ssh_relay = ssh_relay
        self._cmd_receiver = cmd_receiver
        self._max_inflight = max_inflight

        self._cmd_stream: _CommandStream | None = None
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

        # Async-side state
        self._sem: asyncio.Semaphore | None = None
        self._worker_locks: dict[str, asyncio.Lock] = {}
        self._inflight: set[futures.Future[CommandResponse]] = set()

    def start(self) -> None:
        if self._thread is not None:
            self.logger.warning("Command listener already started")
            return
        if self._cmd_stream is not None:
            self.logger.warning("Command stream is already initialized")
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            self.logger.error(
                "Command listener must be started inside an event loop: %s", exc
            )
            return
        assert not self._running
        self._running = True
        self._sem = asyncio.Semaphore(self._max_inflight)
        self._worker_locks = {}
        self._inflight = set()
        self._cmd_stream = _CommandStream(
            node_id=self._node_id,
            redis=self._redis,
            cmd_receiver=self._cmd_receiver,
            logger=self.logger,
        )
        self._thread = Thread(
            target=self._run,
            name="CommandListenerThread",
            daemon=True,
        )
        self._thread.start()
        self.logger.info(
            "Command listener started (max_inflight=%d)", self._max_inflight
        )

    def rebind(self, node_id: str) -> None:
        """Request moving the command subscription to a new node id.

        The reader thread applies the switch; ``wait_rebound`` blocks until it has taken
        effect.
        """
        self._node_id = node_id
        if stream := self._cmd_stream:
            stream.rebind(node_id)

    def wait_rebound(self, timeout: float) -> bool:
        return stream.wait_rebound(timeout) if (stream := self._cmd_stream) else True

    async def stop(self) -> None:
        if self._thread is None or self._cmd_stream is None:
            self.logger.warning("Command listener not started")
            return
        assert self._running
        self._running = False
        self._cmd_stream.close()
        self._cmd_stream = None
        await asyncio.to_thread(self._thread.join)
        self._thread = None

        await self._drain_inflight()

        self._sem = None
        self._worker_locks = {}
        self._loop = None
        self.logger.info("Command listener stopped")

    async def _drain_inflight(self) -> None:
        if not self._inflight:
            return
        pending = [asyncio.wrap_future(f) for f in self._inflight]
        self.logger.info(
            "Draining %d inflight command(s) (timeout=%.1fs)",
            len(pending),
            _STOP_DRAIN_TIMEOUT,
        )
        _, not_done = await asyncio.wait(pending, timeout=_STOP_DRAIN_TIMEOUT)
        for fut in not_done:
            fut.cancel()
        if not_done:
            self.logger.warning(
                "Cancelled %d command(s) that did not finish within drain timeout",
                len(not_done),
            )
        self._inflight.clear()

    def _run(self) -> None:
        cmd_stream = self._cmd_stream
        loop = self._loop
        if cmd_stream is None or loop is None:
            self.logger.error("Command listener not properly initialized")
            return
        try:
            for cmd, resp_handler in cmd_stream.iter_stream():
                fut = asyncio.run_coroutine_threadsafe(self._dispatch(cmd), loop)
                self._inflight.add(fut)
                fut.add_done_callback(self._make_done_callback(cmd, resp_handler))
        except Exception as exc:
            if self._running:
                self.logger.error("Command listener loop error: %s", exc)

    def _make_done_callback(
        self, cmd: CommandMessage, resp_handler: ResponseHandler
    ) -> Callable[[futures.Future[CommandResponse]], None]:
        def on_done(fut: futures.Future[CommandResponse]) -> None:
            self._inflight.discard(fut)
            try:
                resp = fut.result()
            except futures.CancelledError:
                resp = CommandResponse.error(cmd, "Command cancelled during shutdown")
            except Exception as exc:
                self.logger.exception("Command dispatch raised: %s", cmd.command)
                resp = CommandResponse.error(cmd, f"Dispatch failed: {exc}")
            loop = self._loop
            if loop is None or loop.is_closed():
                # Loop is gone (shutdown). Send synchronously so we don't drop
                # the response on the floor for the IPC-side caller.
                self._safe_call_response_handler(resp_handler, resp)
                return
            loop.run_in_executor(
                None, self._safe_call_response_handler, resp_handler, resp
            )

        return on_done

    def _safe_call_response_handler(
        self, resp_handler: ResponseHandler, resp: CommandResponse
    ) -> None:
        try:
            resp_handler(resp)
        except Exception:
            self.logger.exception("Failed to deliver command response")

    # ------------------------------------------------------------------ #
    # Dispatch — semaphore-bounded, per-worker-serialized
    # ------------------------------------------------------------------ #

    async def _dispatch(self, cmd: CommandMessage) -> CommandResponse:
        sem = self._sem
        if sem is None:
            return CommandResponse.error(cmd, "Command listener not initialized")
        try:
            async with sem:
                names = self._target_worker_names(cmd)
                if not names:
                    return await self._invoke(cmd)
                async with contextlib.AsyncExitStack() as stack:
                    for name in names:
                        if name in self._worker_locks:
                            lock = self._worker_locks[name]
                        else:
                            lock = asyncio.Lock()
                            self._worker_locks[name] = lock
                        await stack.enter_async_context(lock)
                    return await self._invoke(cmd)
        except Exception as exc:
            self.logger.exception("Command dispatch failed: %s", cmd.command)
            return CommandResponse.error(cmd, f"Dispatch failed: {exc}")

    @staticmethod
    def _target_worker_names(cmd: CommandMessage) -> list[str]:
        """Names whose per-worker lock the dispatch must hold for FIFO
        ordering against concurrent ops on the same worker. CREATE
        commands generate names internally, so locking them is not
        meaningful (the registry already rejects collisions atomically).
        """
        payload = cmd.payload or {}
        match cmd.command:
            case (
                CommandType.START_WORKER
                | CommandType.STOP_WORKER
                | CommandType.DESTROY_WORKER
            ):
                name = payload.get("worker_name")
                return [] if name is None else [str(name)]
            case CommandType.DESTROY_WORKERS:
                raw = payload.get("worker_names")
                return [] if raw is None else sorted(set(str(n) for n in raw))
            case _:
                return []

    async def _invoke(self, cmd: CommandMessage) -> CommandResponse:
        match cmd.command:
            case CommandType.START_WORKER:
                return await self._handle_start_worker_cmd(cmd)
            case CommandType.CREATE_WORKER:
                return await self._handle_create_worker_cmd(cmd)
            case CommandType.CREATE_WORKER_ON_NODE:
                return await self._handle_create_worker_on_node_cmd(cmd)
            case CommandType.GET_WORKERS:
                return self._handle_get_workers_cmd(cmd)
            case CommandType.STOP_WORKER:
                return await self._handle_stop_worker_cmd(cmd)
            case CommandType.DESTROY_WORKER:
                return await self._handle_destroy_worker_cmd(cmd)
            case CommandType.DESTROY_WORKERS:
                return await self._handle_destroy_workers_cmd(cmd)
            case CommandType.START_SSH_RELAY:
                return self._handle_start_ssh_relay_cmd(cmd)
            case _:
                return CommandResponse.error(cmd, f"Unknown command: {cmd.command}")

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    async def _handle_start_worker_cmd(self, cmd: CommandMessage) -> CommandResponse:
        if cmd.payload is None:
            return CommandResponse.error(
                cmd, "Missing payload for START_WORKER command"
            )

        worker_name = cmd.payload.get("worker_name")
        if worker_name is None:
            return CommandResponse.error(
                cmd, "Missing worker_name in payload for START_WORKER command"
            )

        try:
            result = await asyncio.wait_for(
                self._wm.start_worker(worker_name), timeout=_START_WORKER_TIMEOUT
            )
            return CommandResponse.ok(cmd, data={"success": result})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to start worker: {exc}")

    async def _handle_stop_worker_cmd(self, cmd: CommandMessage) -> CommandResponse:
        if cmd.payload is None:
            return CommandResponse.error(cmd, "Missing payload for STOP_WORKER command")

        worker_name = cmd.payload.get("worker_name")
        if worker_name is None:
            return CommandResponse.error(
                cmd, "Missing worker_name in payload for STOP_WORKER command"
            )

        try:
            result = await asyncio.wait_for(
                self._wm.stop_worker(worker_name), timeout=_STOP_WORKER_TIMEOUT
            )
            return CommandResponse.ok(cmd, data={"success": result})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to stop worker: {exc}")

    def _handle_get_workers_cmd(self, cmd: CommandMessage) -> CommandResponse:
        try:
            if payload := cmd.payload:
                if (worker_name := payload.get("worker_name")) is None:
                    return CommandResponse.error(
                        cmd, "Missing worker_name in payload for GET_WORKERS command"
                    )
                workers = (
                    [worker]
                    if (worker := self._wm.get_worker_info(worker_name))
                    else []
                )
            else:
                workers = self._wm.list_workers()
            return CommandResponse.ok(
                cmd, data={"workers": [worker.model_dump() for worker in workers]}
            )
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to get workers: {exc}")

    def _handle_start_ssh_relay_cmd(self, cmd: CommandMessage) -> CommandResponse:
        if self._ssh_relay is None:
            return CommandResponse.error(cmd, "SSH relay service not available")
        if cmd.payload is None:
            return CommandResponse.error(
                cmd, "Missing payload for START_SSH_RELAY command"
            )
        relay_token = cmd.payload.get("relay_token")
        target_host = cmd.payload.get("target_host")
        target_port = cmd.payload.get("target_port")
        session_id = cmd.payload.get("session_id")
        if not (relay_token and target_host and target_port and session_id):
            return CommandResponse.error(
                cmd,
                "Missing relay_token, target_host, target_port, or session_id "
                "for START_SSH_RELAY command",
            )
        try:
            self._ssh_relay.start_uplink(
                self._redis.telemetry_client,
                str(relay_token),
                str(target_host),
                int(target_port),
                str(session_id),
            )
            return CommandResponse.ok(cmd)
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to start SSH relay: {exc}")

    async def _handle_create_worker_cmd(self, cmd: CommandMessage) -> CommandResponse:
        """Handle CREATE_WORKER: payload is a WorkerInitConfig dict."""
        try:
            init_config = WorkerInitConfig.model_validate(cmd.payload)
            info = await asyncio.wait_for(
                self._wm.create_worker(init_config), timeout=_CREATE_WORKER_TIMEOUT
            )
            return CommandResponse.ok(cmd, data=info.model_dump())
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to create worker: {exc}")

    async def _handle_create_worker_on_node_cmd(
        self, cmd: CommandMessage
    ) -> CommandResponse:
        """Handle CREATE_WORKER_ON_NODE: DockerWorkerConfig payload with GPU
        allocation hint. The factory atomically reserves GPUs."""
        try:
            payload = (cmd.payload or {}).copy()

            gpu_count = int(payload.pop("gpu_count", 0))
            if gpu_count > 0:
                worker_type = payload.get("worker_type")
                if worker_type is None:
                    payload["worker_type"] = "gpu"
                elif worker_type != "gpu":
                    return CommandResponse.error(
                        cmd,
                        f"Invalid worker_type {worker_type} for gpu_count > 0; "
                        "must be 'gpu'",
                    )
                cuda_devices = payload.get("cuda_devices")
                if cuda_devices is None:
                    payload["gpu_count"] = gpu_count
                elif len(cuda_devices) != gpu_count:
                    return CommandResponse.error(
                        cmd,
                        f"Length of cuda_devices list must match gpu_count; got "
                        f"{len(cuda_devices)} devices for gpu_count {gpu_count}",
                    )

            if not payload.get("worker_alias"):
                payload["worker_alias"] = (
                    f"worker_gpu_{secrets.token_hex(6)}"
                    if gpu_count > 0
                    else f"worker_cpu_{secrets.token_hex(6)}"
                )

            worker_config = DockerWorkerConfig(**payload)
            init_config = WorkerInitConfig(
                provider="docker",
                init_on_start=True,
                worker_config=worker_config.model_dump(exclude_unset=True),
            )
            info = await asyncio.wait_for(
                self._wm.create_worker(init_config), timeout=_CREATE_WORKER_TIMEOUT
            )
            return CommandResponse.ok(cmd, data={"worker_name": info.name})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to create worker: {exc}")

    async def _handle_destroy_worker_cmd(self, cmd: CommandMessage) -> CommandResponse:
        worker_name = (cmd.payload or {}).get("worker_name")
        if not worker_name:
            return CommandResponse.error(cmd, "Missing worker_name")
        try:
            success = await asyncio.wait_for(
                self._wm.destroy_worker(worker_name), timeout=_DESTROY_WORKER_TIMEOUT
            )
            self._worker_locks.pop(worker_name, None)
            return CommandResponse.ok(cmd, data={"success": success})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to destroy worker: {exc}")

    async def _handle_destroy_workers_cmd(self, cmd: CommandMessage) -> CommandResponse:
        raw_names = (cmd.payload or {}).get("worker_names")
        names: set[str] | None = set(raw_names) if raw_names is not None else None
        try:
            await asyncio.wait_for(
                self._wm.destroy_workers(names), timeout=_DESTROY_WORKERS_TIMEOUT
            )
            if names is not None:
                for name in names:
                    self._worker_locks.pop(name, None)
            else:
                self._worker_locks.clear()
            return CommandResponse.ok(cmd)
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to destroy workers: {exc}")
