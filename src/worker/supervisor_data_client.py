"""Worker-side stub for the supervisor's data-cache RPCs.

The supervisor owns the node-local Redis connection. Workers ask the
supervisor over gRPC (``FetchData`` / ``PublishData``) instead of opening
their own Redis client, so a single Redis credential lives on the
supervisor and worker images don't need a Redis driver at all.

Each call opens an insecure or TLS gRPC channel pointed at the supervisor
(localhost in single-node deployments). Channels are kept alive between
calls per ``SupervisorDataClient`` instance; close via ``close()``.
"""

import base64
import binascii
import logging
from typing import Any

import grpc

from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc

logger = logging.getLogger(__name__)

_GRPC_MAX_MSG_BYTES = 1024 * 1024 * 1024  # 1 GB; matches supervisor server.


class SupervisorDataClient:
    """Connects to the supervisor and forwards `FetchData` / `PublishData`."""

    def __init__(
        self,
        grpc_target: str,
        worker_token: str,
        grpc_tls_ca_b64: str | None = None,
    ) -> None:
        self._grpc_target = grpc_target
        self._worker_token = worker_token
        self._grpc_tls_ca_b64 = grpc_tls_ca_b64
        self._channel: grpc.Channel | None = None
        self._stub: supervisor_pb2_grpc.SupervisorStub | None = None

    def fetch(self, data_id: str) -> bytes | None:
        """Return cached payload bytes, or None on miss / error."""
        stub = self._ensure_stub()
        request = supervisor_pb2.FetchDataRequest(data_id=data_id)
        try:
            resp = stub.FetchData(request, metadata=self._metadata())
        except grpc.RpcError as exc:
            logger.warning(
                "Supervisor FetchData(%s) failed; treating as cache miss: %s",
                data_id,
                exc.details() if hasattr(exc, "details") else exc,
            )
            return None
        if not resp.found:
            return None
        return bytes(resp.payload)

    def publish(self, data_id: str, payload: bytes, ttl_sec: int) -> bool:
        """Best-effort publish. Returns True if the supervisor confirmed write."""
        stub = self._ensure_stub()
        request = supervisor_pb2.PublishDataRequest(
            data_id=data_id, payload=payload, ttl_sec=max(0, int(ttl_sec))
        )
        try:
            resp = stub.PublishData(request, metadata=self._metadata())
        except grpc.RpcError as exc:
            logger.warning(
                "Supervisor PublishData(%s) failed; durable HTTP upload remains "
                "the source of truth: %s",
                data_id,
                exc.details() if hasattr(exc, "details") else exc,
            )
            return False
        return bool(resp.ok)

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self._worker_token}"),)

    def _ensure_stub(self) -> supervisor_pb2_grpc.SupervisorStub:
        if self._stub is not None:
            return self._stub
        options: list[tuple[str, Any]] = [
            ("grpc.max_receive_message_length", _GRPC_MAX_MSG_BYTES),
            ("grpc.max_send_message_length", _GRPC_MAX_MSG_BYTES),
        ]
        root_cert = self._load_tls_root_cert()
        if root_cert is not None:
            creds = grpc.ssl_channel_credentials(root_certificates=root_cert)
            self._channel = grpc.secure_channel(
                self._grpc_target, creds, options=options
            )
        else:
            self._channel = grpc.insecure_channel(self._grpc_target, options=options)
        self._stub = supervisor_pb2_grpc.SupervisorStub(self._channel)
        return self._stub

    def _load_tls_root_cert(self) -> bytes | None:
        if not self._grpc_tls_ca_b64:
            return None
        try:
            return base64.b64decode(self._grpc_tls_ca_b64)
        except (ValueError, binascii.Error) as exc:
            logger.warning("Invalid SUPERVISOR_GRPC_TLS_CA_B64: %s", exc)
            return None
