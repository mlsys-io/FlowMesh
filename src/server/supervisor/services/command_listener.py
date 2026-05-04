import asyncio
import logging
import queue
import secrets
from collections.abc import Callable, Iterable
from threading import Thread

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
)
from ...utils.concurrent import Sentinel, TaskReceiver
from ...utils.helpers import iter_pubsub_messages
from ..adapters.docker import DockerWorkerConfig
from ..manager import WorkerInitConfig, WorkerManager
from ..resource_manager import ResourceManager
from .ssh_relay import SshRelayService

type ResponseHandler = Callable[[CommandResponse], None]


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


def _pubsub_loop(
    redis: SyncRedisClient,
    pubsub: PubSub,
    q: queue.Queue[tuple[CommandMessage, ResponseHandler]],
    logger: logging.Logger,
) -> None:
    for data in iter_pubsub_messages(pubsub):
        try:
            cmd = CommandMessage.model_validate(data)
        except ValidationError as e:
            logger.error("Invalid command message: %s", e)
            continue

        def send_response(resp: CommandResponse) -> None:
            redis.publish_control(NODE_RESPONSE_CHANNEL, resp.model_dump_json())

        q.put((cmd, send_response))


class _CommandStream:
    _SENTINEL = Sentinel()

    def __init__(
        self,
        node_id: str,
        redis: SyncRedisClient,
        cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None,
        logger: logging.Logger,
    ):
        self.node_id = node_id
        self.redis = redis
        self.cmd_receiver = cmd_receiver
        self.logger = logger
        self._cmd_queue: (
            queue.Queue[tuple[CommandMessage, ResponseHandler] | Sentinel] | None
        ) = None

    def close(self) -> None:
        if self._cmd_queue is not None:
            self._cmd_queue.put(self._SENTINEL)
            self._cmd_queue = None

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
        pubsub_thread = Thread(
            target=_pubsub_loop,
            args=(self.redis, pubsub, cmd_queue, self.logger),
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
            pubsub.close()


class CommandListener:
    def __init__(
        self,
        redis: SyncRedisClient,
        node_id: str,
        worker_manager: WorkerManager,
        logger: logging.Logger,
        cmd_receiver: TaskReceiver[CommandMessage, CommandResponse] | None = None,
        ssh_relay: SshRelayService | None = None,
    ) -> None:
        self.logger = logger
        self._redis = redis
        self._node_id = node_id
        self._wm = worker_manager
        self._ssh_relay = ssh_relay
        self._cmd_receiver = cmd_receiver

        self._cmd_stream: _CommandStream | None = None
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

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
        self.logger.info("Command listener started")

    def stop(self) -> None:
        if self._thread is None or self._cmd_stream is None:
            self.logger.warning("Command listener not started")
            return
        assert self._running
        self._running = False
        self._cmd_stream.close()
        self._cmd_stream = None
        self._thread.join()
        self._thread = None
        self._loop = None
        self.logger.info("Command listener stopped")

    def _run(self) -> None:
        cmd_stream = self._cmd_stream
        loop = self._loop
        if cmd_stream is None or loop is None:
            self.logger.error("Command listener not properly initialized")
            return
        try:
            resp: CommandResponse | None
            for cmd, resp_handler in cmd_stream.iter_stream():
                match cmd.command:
                    case CommandType.START_WORKER:
                        resp = self._handle_start_worker_cmd(cmd, loop)
                    case CommandType.CREATE_WORKER:
                        resp = self._handle_create_worker_cmd(cmd, loop)
                    case CommandType.CREATE_WORKER_ON_NODE:
                        resp = self._handle_create_worker_on_node_cmd(cmd, loop)
                    case CommandType.GET_WORKERS:
                        resp = self._handle_get_workers_cmd(cmd)
                    case CommandType.STOP_WORKER:
                        resp = self._handle_stop_worker_cmd(cmd, loop)
                    case CommandType.DESTROY_WORKER:
                        resp = self._handle_destroy_worker_cmd(cmd, loop)
                    case CommandType.DESTROY_WORKERS:
                        resp = self._handle_destroy_workers_cmd(cmd, loop)
                    case CommandType.START_SSH_RELAY:
                        resp = self._handle_start_ssh_relay_cmd(cmd)
                    case _:
                        resp = CommandResponse.error(
                            cmd, f"Unknown command: {cmd.command}"
                        )

                if resp is not None:
                    resp_handler(resp)

        except Exception as exc:
            if self._running:
                self.logger.error("Command listener loop error: %s", exc)

    def _handle_start_worker_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
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
            result = asyncio.run_coroutine_threadsafe(
                self._wm.start_worker(worker_name), loop
            ).result()
            return CommandResponse.ok(cmd, data={"success": result})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to start worker: {exc}")

    def _handle_stop_worker_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
        if cmd.payload is None:
            return CommandResponse.error(cmd, "Missing payload for STOP_WORKER command")

        worker_name = cmd.payload.get("worker_name")
        if worker_name is None:
            return CommandResponse.error(
                cmd, "Missing worker_name in payload for STOP_WORKER command"
            )

        try:
            result = asyncio.run_coroutine_threadsafe(
                self._wm.stop_worker(worker_name), loop
            ).result()
            return CommandResponse.ok(cmd, data={"success": result})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to stop worker: {exc}")

    def _handle_get_workers_cmd(self, cmd: CommandMessage) -> CommandResponse:
        try:
            workers = self._wm.list_workers()
            workers_data = [worker.model_dump() for worker in workers]
            return CommandResponse.ok(cmd, data={"workers": workers_data})
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

    def _handle_create_worker_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
        """Handle CREATE_WORKER: payload is a WorkerInitConfig dict."""
        try:
            init_config = WorkerInitConfig.model_validate(cmd.payload)
            info = asyncio.run_coroutine_threadsafe(
                self._wm.create_worker(init_config), loop
            ).result(timeout=600.0)
            return CommandResponse.ok(cmd, data=info.model_dump())
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to create worker: {exc}")

    def _handle_create_worker_on_node_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
        """Handle CREATE_WORKER_ON_NODE: DockerWorkerConfig payload with GPU
        allocation hint."""
        try:
            payload = (cmd.payload or {}).copy()

            gpu_count = int(payload.pop("gpu_count", 0))
            if gpu_count > 0:
                rm = ResourceManager.get_instance()
                # Validate or set worker_type
                worker_type = payload.get("worker_type")
                if worker_type is None:
                    payload["worker_type"] = "gpu"
                elif worker_type != "gpu":
                    return CommandResponse.error(
                        cmd,
                        f"Invalid worker_type {worker_type} for gpu_count > 0; "
                        "must be 'gpu'",
                    )
                # Validate or set cuda_devices
                cuda_devices = payload.get("cuda_devices")
                if cuda_devices is None:
                    # Preselect N GPUs; factory will call allocate_gpus() on them.
                    # Raises ValueError if fewer than gpu_count GPUs are available.
                    payload["cuda_devices"] = rm.next_available_gpus(gpu_count)
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
            info = asyncio.run_coroutine_threadsafe(
                self._wm.create_worker(init_config), loop
            ).result(timeout=600.0)
            return CommandResponse.ok(cmd, data={"worker_name": info.name})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to create worker: {exc}")

    def _handle_destroy_worker_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
        worker_name = (cmd.payload or {}).get("worker_name")
        if not worker_name:
            return CommandResponse.error(cmd, "Missing worker_name")
        try:
            success = asyncio.run_coroutine_threadsafe(
                self._wm.destroy_worker(worker_name), loop
            ).result(timeout=60.0)
            return CommandResponse.ok(cmd, data={"success": success})
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to destroy worker: {exc}")

    def _handle_destroy_workers_cmd(
        self, cmd: CommandMessage, loop: asyncio.AbstractEventLoop
    ) -> CommandResponse:
        raw_names = (cmd.payload or {}).get("worker_names")
        names: set[str] | None = set(raw_names) if raw_names is not None else None
        try:
            asyncio.run_coroutine_threadsafe(
                self._wm.destroy_workers(names), loop
            ).result(timeout=120.0)
            return CommandResponse.ok(cmd)
        except Exception as exc:
            return CommandResponse.error(cmd, f"Failed to destroy workers: {exc}")
