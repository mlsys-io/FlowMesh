import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import grpc
import grpc.aio
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from shared.grpc.supervisor.v1 import (
    supervisor_pb2,
    supervisor_pb2_grpc,
)
from shared.utils import new_worker_id

from ... import env
from ...clients.redis import (
    WORKER_ID_SEQ_KEY,
    WORKERS_SET_KEY,
    SyncRedisClient,
    worker_key,
)
from ..adapters.base import WorkerAdapter, WorkerTokenType
from ..registry import WorkerRegistry
from ..schemas import WorkerStatus
from ..services.relay_service import RelayService
from ..services.task_listener import TaskListener


def _token_from_metadata(metadata: Iterable[tuple[str, str]]) -> WorkerTokenType | None:
    for key, value in metadata:
        key_l = key.lower()
        if key_l == "authorization" and value.startswith("Bearer "):
            return value[7:]  # type: ignore
        if key_l == "x-worker-token":
            return value  # type: ignore
    return None


def _load_tls_credentials() -> grpc.ServerCredentials | None:
    cert_file = env.SERVER_GRPC_TLS_CERT_FILE
    key_file = env.SERVER_GRPC_TLS_KEY_FILE
    if not (cert_file or key_file):
        return None
    if not (cert_file and key_file):
        raise RuntimeError(
            "SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE are required"
        )
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    try:
        cert_bytes = cert_path.read_bytes()
        key_bytes = key_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Failed to read server TLS files: {exc}") from exc
    return grpc.ssl_server_credentials([(key_bytes, cert_bytes)])


def _struct_from_payload(payload: dict) -> Struct:
    struct = Struct()
    struct.update(payload)
    return struct


def _payload_from_struct(struct: Struct) -> dict:
    return MessageToDict(struct, preserving_proto_field_name=True)


class SupervisorServicer(supervisor_pb2_grpc.SupervisorServicer):
    def __init__(
        self,
        registry: WorkerRegistry,
        redis: SyncRedisClient,
        node_id: str,
        node_alias: str,
        task_listener: TaskListener,
        relay_service: RelayService,
        logger: logging.Logger,
    ) -> None:
        self._registry = registry
        self._task_listener = task_listener
        self._relay_service = relay_service
        self._redis = redis
        self._node_id = node_id
        self._node_alias = node_alias
        self._logger = logger

    def rebind_node(self, node_id: str) -> None:
        """Re-home this node's workers under a new node id.

        Future registrations stamp the new id, and every already-registered
        worker's ``node_id`` field is rewritten in Redis so the dispatcher
        routes tasks to them on the node's new dispatch channel.
        """
        if node_id == self._node_id:
            return
        old_node_id = self._node_id
        self._node_id = node_id
        rehomed = 0
        for worker in self._registry.all_workers():
            worker_id = self._registry.get_worker_id(worker.token)
            if worker_id is None:
                continue
            self._redis.hash_set(worker_key(worker_id), {"node_id": node_id})
            rehomed += 1
        self._logger.info(
            "Re-homed %d worker(s) from node %s to %s",
            rehomed,
            old_node_id,
            node_id,
        )

    async def RegisterWorker(
        self,
        request: supervisor_pb2.RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> supervisor_pb2.RegisterResponse:
        worker = self._get_worker_from_context(context)
        if worker is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")
        worker_meta = _payload_from_struct(request.meta)
        worker_id = new_worker_id(self._redis.incr(WORKER_ID_SEQ_KEY))
        worker_meta["id"] = worker_id
        worker_meta["node_id"] = self._node_id
        worker_meta["node_alias"] = self._node_alias
        self._redis.sadd(WORKERS_SET_KEY, worker_id)
        self._redis.hash_set(worker_key(worker_id), worker_meta)
        self._registry.set_worker_id(worker.token, worker_id)
        self._task_listener.add_worker(worker_id)
        try:
            worker.set_worker_id(worker_id)
        except RuntimeError as exc:
            self._logger.warning(exc)
        self._logger.info("Registered worker %s", worker_id)
        return supervisor_pb2.RegisterResponse(worker_id=worker_id)

    async def StreamTasks(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[supervisor_pb2.DispatchMessage]:
        worker_id = self._get_worker_id_from_context(context)
        if worker_id is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        while True:
            try:
                event = await self._task_listener.get_event(worker_id)
            except asyncio.CancelledError:
                break
            if event.get("kind") == "interrupt":
                yield supervisor_pb2.DispatchMessage(
                    interrupt=supervisor_pb2.InterruptMessage(
                        task_id=str(event["task_id"]),
                        reason=str(event["reason"]),
                    )
                )
            elif event.get("kind") == "stop":
                yield supervisor_pb2.DispatchMessage(
                    stop=supervisor_pb2.StopMessage(
                        task_id=str(event["task_id"]),
                        reason=str(event["reason"]),
                    )
                )
            else:
                yield supervisor_pb2.DispatchMessage(
                    task=supervisor_pb2.TaskMessage(payload=_struct_from_payload(event))
                )
        self._logger.info("Task stream closed for worker %s", worker_id)

    async def PushEvents(
        self,
        request_iterator: AsyncIterator[supervisor_pb2.EventMessage],
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        worker = self._get_worker_from_context(context)
        if worker is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        worker_id = self._registry.get_worker_id(worker.token)
        if worker_id is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Worker not registered"
            )

        registered: bool = False
        unregistered: bool = False
        async for message in request_iterator:
            payload = _payload_from_struct(message.payload)
            # Trap register/unregister events
            match payload.get("type"):
                case "REGISTER":
                    registered = True
                    worker.set_status(WorkerStatus.RUNNING)
                case "UNREGISTER":
                    unregistered = True
            self._relay_service.add_event(payload)
        self._logger.info("Event stream closed for worker %s", worker_id)
        if registered and not unregistered:
            # Manually send unregister event if not sent by worker
            payload = dict(type="UNREGISTER", worker_id=worker_id, payload={})
            self._relay_service.add_event(payload)
        try:
            worker.clear_worker_id()
        except RuntimeError as exc:
            self._logger.warning(exc)
        worker.set_status(WorkerStatus.STOPPED)
        return Empty()

    async def PushLogs(
        self,
        request_iterator: AsyncIterator[supervisor_pb2.LogMessage],
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        worker_id = self._get_worker_id_from_context(context)
        if worker_id is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        async for msg in request_iterator:
            payload = _payload_from_struct(msg.payload)
            if isinstance(payload, dict):
                payload.setdefault("worker_id", worker_id)
            self._relay_service.add_log(payload)
        self._logger.debug("Log stream closed for worker %s", worker_id)
        return Empty()

    def _get_worker_from_context(
        self, context: grpc.aio.ServicerContext
    ) -> WorkerAdapter | None:
        token = _token_from_metadata(context.invocation_metadata())  # type: ignore
        if token is None:
            return None
        return self._registry.try_get(token)

    def _get_worker_id_from_context(
        self, context: grpc.aio.ServicerContext
    ) -> str | None:
        token = _token_from_metadata(context.invocation_metadata())  # type: ignore
        if token is None:
            return None
        return self._registry.get_worker_id(token)


_GRPC_MAX_MSG_BYTES = 1024 * 1024 * 1024  # 1 GB


class GrpcServer:
    def __init__(
        self,
        host: str,
        port: int,
        registry: WorkerRegistry,
        redis: SyncRedisClient,
        node_id: str,
        node_alias: str,
        task_listener: TaskListener,
        relay_service: RelayService,
        logger: logging.Logger,
    ) -> None:
        self._logger = logger
        self._server: grpc.aio.Server | None = None
        self._servicer = SupervisorServicer(
            registry, redis, node_id, node_alias, task_listener, relay_service, logger
        )
        self._listen_addr = f"{host}:{port}"

    def rebind_node(self, node_id: str) -> None:
        """Re-home registered workers under a new node id."""
        self._servicer.rebind_node(node_id)

    async def start(self) -> None:
        if self._server is not None:
            self._logger.warning("Server gRPC server already started")
            return
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", _GRPC_MAX_MSG_BYTES),
                ("grpc.max_send_message_length", _GRPC_MAX_MSG_BYTES),
                (
                    "grpc.keepalive_permit_without_calls",
                    int(env.SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS),
                ),
                (
                    "grpc.http2.min_recv_ping_interval_without_data_ms",
                    env.SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS,
                ),
            ]
        )
        supervisor_pb2_grpc.add_SupervisorServicer_to_server(
            self._servicer, self._server
        )
        creds = (
            None if env.SUPERVISOR_GRPC_DISABLE_SERVER_TLS else _load_tls_credentials()
        )
        if creds is None:
            bound_port = self._server.add_insecure_port(self._listen_addr)
            self._logger.warning(
                "Server gRPC TLS disabled; running insecure on %s",
                self._listen_addr,
            )
        else:
            bound_port = self._server.add_secure_port(self._listen_addr, creds)
            self._logger.info(
                "Server gRPC TLS enabled; running secure on %s", self._listen_addr
            )
        if not bound_port:
            raise RuntimeError(f"Failed to bind gRPC server to {self._listen_addr}")
        await self._server.start()
        self._logger.info("Server gRPC server started on %s", self._listen_addr)

    async def stop(self, grace: float = 5.0) -> None:
        if self._server is None:
            return
        await self._server.stop(grace)
        self._server = None
        self._logger.info("Server gRPC server stopped")
