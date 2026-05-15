import base64
import binascii
import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc
from shared.schemas.event import Event, TaskEvent, WorkerEvent, serialize_event
from shared.schemas.worker import SSHLimits
from shared.tasks.worker_message import (
    WorkerHardware,
    WorkerStatus,
    WorkerTaskMessage,
)
from shared.utils.json import normalize_numbers
from shared.utils.time import now_iso

from .utils.logging import TaskLogEmitter


class SupervisorClient:
    """Bidirectional transport that lets the worker talk to its supervisor."""

    _TASK_SENTINEL = object()
    _EVENT_SENTINEL = object()
    _GRPC_MAX_MSG_BYTES = 1024 * 1024 * 1024  # 1 GB

    def __init__(
        self,
        worker_token: str,
        owner_principal: dict[str, Any] | None,
        grpc_target: str,
        worker_namespace: str,
        worker_cluster: str,
        worker_alias: str,
        logger: logging.Logger,
        grpc_tls_ca_b64: str | None = None,
        grpc_keepalive_time_ms: int | None = None,
        grpc_keepalive_timeout_ms: int | None = None,
    ):
        self.owner_principal = owner_principal
        self.grpc_target = grpc_target
        self.worker_namespace = worker_namespace
        self.worker_cluster = worker_cluster
        self.worker_alias = worker_alias
        self.logger = logger
        self._worker_token = worker_token
        self._grpc_tls_ca_b64 = grpc_tls_ca_b64
        self._grpc_keepalive_time_ms = grpc_keepalive_time_ms
        self._grpc_keepalive_timeout_ms = grpc_keepalive_timeout_ms

        self._worker_id: str | None = None
        self._worker_register_event: WorkerEvent | None = None
        self._drain = threading.Event()
        self._shutdown = threading.Event()
        self._shutdown.set()  # Initially shutdown
        self._stop = threading.Event()
        self._stop.set()  # Initially stopped
        self._event_ready = threading.Event()
        self._task_ready = threading.Event()
        self._task_queue: queue.Queue[WorkerTaskMessage | object] = queue.Queue()
        self._interrupt_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._event_queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._event_thread: threading.Thread | None = None
        self._task_thread: threading.Thread | None = None
        self._channel: grpc.Channel | None = None
        self._stub: supervisor_pb2_grpc.SupervisorStub | None = None

    @property
    def worker_id(self) -> str:
        if self._worker_id is None:
            raise RuntimeError("Worker not registered with supervisor")
        return self._worker_id

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def register(
        self,
        status: WorkerStatus,
        started_at: str,
        pid: int,
        env: dict[str, Any],
        hardware: WorkerHardware,
        ssh_limits: SSHLimits | None,
        tags: list[str],
        cost_per_hour: float,
        power_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Register the worker with the supervisor.

        This must be called before the client is started.
        """
        if self._worker_id is not None:
            self.logger.warning("Worker already registered")
            return
        worker_meta = {
            "namespace": self.worker_namespace,
            "cluster": self.worker_cluster,
            "alias": self.worker_alias,
            "status": status.value,
            "started_at": started_at,
            "pid": str(pid),
            "env_json": json.dumps(env, ensure_ascii=False),
            "hardware_json": hardware.model_dump_json(),
            "tags_json": json.dumps(tags, ensure_ascii=False),
            "last_seen": started_at,
            "cost_per_hour": str(cost_per_hour),
        }
        if ssh_limits is not None:
            worker_meta["ssh_limits_json"] = ssh_limits.model_dump_json()
        self._register_grpc(worker_meta)
        self.logger.info("Worker connected via supervisor at %s", self.grpc_target)

        # Create a temporary register event for later use
        payload: dict[str, Any] = {
            "env": env,
            "hardware": hardware.model_dump(mode="python"),
            "cost_per_hour": cost_per_hour,
        }
        if power_metrics:
            payload["power_metrics"] = power_metrics
        self._worker_register_event = WorkerEvent(
            type="REGISTER",
            worker_id=self.worker_id,
            status=status,
            ts=started_at,
            tags=tags,
            payload=payload,
            actor=self.owner_principal,
        )

    def start(self) -> None:
        """Start background threads to handle events and tasks.

        This requires the worker to be registered first.
        """
        if not self._shutdown.is_set():
            self.logger.warning("Supervisor client already started")
            return
        self._shutdown.clear()
        self._stop.clear()
        self._channel = self._create_grpc_channel()
        self._stub = supervisor_pb2_grpc.SupervisorStub(self._channel)
        self._start_event_stream()
        self._start_task_stream()
        self._send_register_event()

    def stop(self) -> None:
        """Stop task pulling without fully shutting down."""
        if self._stop.is_set():
            return
        self._stop.set()
        # Close the task queue while keeping the event queue open
        self._task_queue.put(self._TASK_SENTINEL)

    def shutdown(self) -> None:
        """Close sockets and stop background threads."""
        if self._worker_id is None:
            self.logger.warning("Supervisor client not started")
            return
        self.stop()
        # Drain event queue and close connections
        self._drain.set()
        self._shutdown.set()
        self._event_queue.put(self._EVENT_SENTINEL)
        if self._event_thread:
            self._event_thread.join(timeout=5)
            self._event_thread = None
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
        if self._task_thread:
            self._task_thread.join(timeout=5)
            self._task_thread = None
        self._event_ready.clear()
        self._task_ready.clear()
        self._worker_id = None
        self._drain.clear()

    # ------------------------------------------------------------------ #
    # Worker lifecycle helpers
    # ------------------------------------------------------------------ #

    def heartbeat(
        self,
        ts: str | None = None,
        metrics: dict[str, Any] | None = None,
        ttl_sec: int = 120,
    ) -> None:
        ts = ts or now_iso()
        event = WorkerEvent(
            type="HEARTBEAT",
            worker_id=self.worker_id,
            ts=ts,
            metrics=metrics or {},
            payload={"ttl_sec": ttl_sec},
        )
        self._send_event(event)

    def set_status(
        self, status: WorkerStatus, extra: dict[str, Any] | None = None
    ) -> None:
        event = WorkerEvent(
            type="STATUS",
            worker_id=self.worker_id,
            status=status,
            payload=extra or {},
        )
        self._send_event(event)

    def unregister(
        self,
        cost_per_hour: float | None = None,
        uptime_sec: float | None = None,
        accrued_cost_usd: float | None = None,
        power_summary: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if cost_per_hour is not None:
            payload["cost_per_hour"] = cost_per_hour
        if uptime_sec is not None:
            payload["uptime_sec"] = uptime_sec
        if accrued_cost_usd is not None:
            payload["accrued_cost_usd"] = accrued_cost_usd
        if power_summary is not None:
            payload["power_summary"] = power_summary
        event = WorkerEvent(
            type="UNREGISTER",
            worker_id=self.worker_id,
            payload=payload,
            actor=self.owner_principal,
        )
        self._send_event(event)

    def task_update(self, task_id: str, payload: dict[str, Any]) -> None:
        event = TaskEvent(
            type="TASK_UPDATE",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=payload,
        )
        self._send_event(event)

    def task_failed(
        self, task_id: str, error: str | None, metadata: dict[str, Any] | None = None
    ) -> None:
        event = TaskEvent(
            type="TASK_FAILED",
            worker_id=self.worker_id,
            task_id=task_id,
            error=error,
            payload=metadata or {},
        )
        self._send_event(event)

    def task_succeeded(
        self, task_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        event = TaskEvent(
            type="TASK_SUCCEEDED",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=metadata or {},
        )
        self._send_event(event)

    def task_started(
        self,
        task_id: str,
        task_type: str | None = None,
        dispatched_at: str | None = None,
        started_at: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if task_type is not None:
            payload["taskType"] = task_type
        if dispatched_at is not None:
            payload["dispatched_at"] = dispatched_at
        if started_at is not None:
            payload["started_at"] = started_at
        event = TaskEvent(
            type="TASK_STARTED",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=payload,
        )
        self._send_event(event)

    def task_cancelled(
        self, task_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        event = TaskEvent(
            type="TASK_CANCELLED",
            worker_id=self.worker_id,
            task_id=task_id,
            payload=metadata or {},
        )
        self._send_event(event)

    def create_task_log_emitter(
        self,
        task_id: str,
        workflow_id: str,
        owner_id: str,
        task_refs: list[dict[str, str]] | None = None,
        log_paths: dict[str, Path] | None = None,
    ) -> TaskLogEmitter | None:
        if self._stub is None:
            return None
        return TaskLogEmitter(
            stub=self._stub,
            metadata=self._grpc_metadata(),
            struct_from_payload=self._struct_from_payload,
            logger=self.logger,
            task_id=task_id,
            workflow_id=workflow_id,
            owner_id=owner_id,
            worker_id=self.worker_id,
            task_refs=task_refs,
            log_paths=log_paths,
        )

    # ------------------------------------------------------------------ #
    # Task consumption
    # ------------------------------------------------------------------ #

    def iter_tasks(self) -> Iterable[WorkerTaskMessage]:
        """Yield tasks relayed by the supervisor until shutdown."""
        while not self._stop.is_set():
            try:
                item = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is self._TASK_SENTINEL:
                break
            assert isinstance(item, WorkerTaskMessage)
            yield item

    def iter_interrupts(self) -> Iterable[tuple[str, str]]:
        while True:
            try:
                yield self._interrupt_queue.get_nowait()
            except queue.Empty:
                break

    def iter_stops(self) -> Iterable[tuple[str, str]]:
        while True:
            try:
                yield self._stop_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _send_register_event(self) -> None:
        event = self._worker_register_event
        if event is None:
            raise RuntimeError("Worker not registered with supervisor")
        self._send_event(event)
        self._worker_register_event = None

    def _register_grpc(self, worker_meta: dict[str, Any]) -> None:
        self.logger.info(
            "Registering worker %s with supervisor %s",
            self.worker_alias,
            self.grpc_target,
        )
        with self._create_grpc_channel() as channel:
            stub = supervisor_pb2_grpc.SupervisorStub(channel)
            request = supervisor_pb2.RegisterRequest(
                meta=self._struct_from_payload(worker_meta)
            )
            metadata = self._grpc_metadata()
            try:
                resp = stub.RegisterWorker(request, metadata=metadata)
            except grpc.RpcError as exc:
                self.logger.error("Supervisor registration failed: %s", exc)
                raise SystemExit(1) from exc
        if not resp.worker_id:
            raise SystemExit("Supervisor registration response missing worker_id")
        self._worker_id = resp.worker_id

    def _start_event_stream(self) -> None:
        self._event_ready.clear()
        thread = threading.Thread(
            target=self._run_event_stream,
            name="SupervisorEventStream",
            daemon=True,
        )
        thread.start()
        self._event_thread = thread
        self._event_ready.wait()

    def _start_task_stream(self) -> None:
        self._task_ready.clear()
        thread = threading.Thread(
            target=self._run_task_stream,
            name="SupervisorTaskStream",
            daemon=True,
        )
        thread.start()
        self._task_thread = thread
        self._task_ready.wait()

    def _run_event_stream(self) -> None:
        if self._channel is None or self._stub is None:
            self.logger.error("Supervisor gRPC channel not initialized")
            return
        metadata = self._grpc_metadata()
        while not self._shutdown.is_set() or self._drain.is_set():
            try:
                grpc.channel_ready_future(self._channel).result(timeout=10)
                self._event_ready.set()
                self._stub.PushEvents(self._event_messages(), metadata=metadata)
                if self._shutdown.is_set():
                    break
                self._event_ready.clear()
                self.logger.warning("Event stream closed, retrying in 3 seconds")
                time.sleep(3)
            except grpc.FutureTimeoutError:
                if self._shutdown.is_set():
                    break
                self._event_ready.clear()
                self.logger.warning("Event stream not ready, retrying in 3 seconds")
                time.sleep(3)
            except grpc.RpcError as exc:
                if self._shutdown.is_set():
                    break
                self._event_ready.clear()
                self.logger.error("Supervisor event stream error: %s", exc)
                time.sleep(3)

    def _run_task_stream(self) -> None:
        if self._channel is None or self._stub is None:
            self.logger.error("Supervisor gRPC channel not initialized")
            return
        metadata = self._grpc_metadata()
        while not self._stop.is_set():
            try:
                grpc.channel_ready_future(self._channel).result(timeout=10)
                self._task_ready.set()
                for message in self._stub.StreamTasks(Empty(), metadata=metadata):
                    if message.HasField("interrupt"):
                        self._interrupt_queue.put(
                            (message.interrupt.task_id, message.interrupt.reason)
                        )
                    elif message.HasField("stop"):
                        self._stop_queue.put(
                            (message.stop.task_id, message.stop.reason)
                        )
                    else:
                        payload = self._payload_from_struct(message.task.payload)
                        try:
                            task_message = WorkerTaskMessage.model_validate(payload)
                        except Exception as exc:
                            self.logger.error(
                                "Failed to parse task message payload: %s; error: %s",
                                payload,
                                exc,
                            )
                        else:
                            self._task_queue.put(task_message)
                    if self._stop.is_set():
                        break
                if self._stop.is_set():
                    break
                self._task_ready.clear()
                self.logger.warning("Task stream closed, retrying in 3 seconds")
                time.sleep(3)
            except grpc.FutureTimeoutError:
                if self._stop.is_set():
                    break
                self._task_ready.clear()
                self.logger.warning("Task stream not ready, retrying in 3 seconds")
                time.sleep(3)
            except grpc.RpcError as exc:
                if self._stop.is_set():
                    break
                self._task_ready.clear()
                self.logger.error("Supervisor task stream error: %s", exc)
                time.sleep(3)

    def _event_messages(self) -> Iterable[supervisor_pb2.EventMessage]:
        while not self._shutdown.is_set() or self._drain.is_set():
            try:
                item = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is self._EVENT_SENTINEL:
                break
            assert isinstance(item, dict)
            yield supervisor_pb2.EventMessage(payload=self._struct_from_payload(item))

    # ---- Networking helpers ----------------------------------------- #

    def _grpc_metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self._worker_token}"),)

    def _create_grpc_channel(self) -> grpc.Channel:
        root_cert = self._load_tls_root_cert()
        options: list[tuple[str, int]] = [
            ("grpc.max_receive_message_length", self._GRPC_MAX_MSG_BYTES),
            ("grpc.max_send_message_length", self._GRPC_MAX_MSG_BYTES),
        ]
        if self._grpc_keepalive_time_ms is not None:
            options.extend(
                [
                    ("grpc.keepalive_time_ms", self._grpc_keepalive_time_ms),
                    (
                        "grpc.keepalive_timeout_ms",
                        self._grpc_keepalive_timeout_ms or 10_000,
                    ),
                    ("grpc.keepalive_permit_without_calls", 1),
                ]
            )
        if root_cert is not None:
            creds = grpc.ssl_channel_credentials(root_certificates=root_cert)
            self.logger.info("Worker gRPC TLS enabled for %s", self.grpc_target)
            return grpc.secure_channel(self.grpc_target, creds, options=options)
        self.logger.warning(
            "Worker gRPC TLS disabled; using insecure channel to %s",
            self.grpc_target,
        )
        return grpc.insecure_channel(self.grpc_target, options=options)

    def _load_tls_root_cert(self) -> bytes | None:
        if not self._grpc_tls_ca_b64:
            return None
        try:
            return base64.b64decode(self._grpc_tls_ca_b64)
        except (ValueError, binascii.Error) as exc:
            self.logger.warning("Invalid SUPERVISOR_GRPC_TLS_CA_B64: %s", exc)
        return None

    def _struct_from_payload(self, payload: dict[str, Any]) -> Struct:
        struct = Struct()
        struct.update(payload)
        return struct

    def _payload_from_struct(self, struct: Struct) -> dict[str, Any]:
        payload = MessageToDict(struct, preserving_proto_field_name=True)
        return normalize_numbers(payload)

    def _send_event(self, event: Event) -> None:
        if self._stub is None:
            raise RuntimeError("Supervisor gRPC client not started")
        # Wait until the stream is ready without an explicit timeout.
        if not self._event_ready.wait():
            raise RuntimeError("Supervisor event stream not ready")
        self._event_queue.put(serialize_event(event))
