"""The worker end of a relayed TCP connection."""

import logging
import socket
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

import grpc

from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc

from .registry import EndpointRegistry

if TYPE_CHECKING:
    from ..supervisor_client import SupervisorClient

logger = logging.getLogger(__name__)

_READ_CHUNK = 16384
_CONNECT_TIMEOUT_SEC = 5.0

# Without this the relay channel would share a subchannel with the control
# channel: gRPC pools them by target and options, and both are identical here.
_OWN_CONNECTION = ("grpc.use_local_subchannel_pool", 1)


class RelayClient:
    """Connects a supervisor's relay request to a local port this worker owns.

    Each request gets its own gRPC stream and its own loopback connection, so
    the relay carries no connection multiplexing of its own.
    """

    def __init__(self, client: "SupervisorClient", endpoints: EndpointRegistry):
        self._client = client
        self._endpoints = endpoints
        self._channel: grpc.Channel | None = None
        self._stub: supervisor_pb2_grpc.SupervisorStub | None = None
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._closing = threading.Event()

    def start(self) -> None:
        self._closing.clear()
        self._channel = self._client.create_grpc_channel(
            extra_options=[_OWN_CONNECTION]
        )
        self._stub = supervisor_pb2_grpc.SupervisorStub(self._channel)

    def shutdown(self) -> None:
        self._closing.set()
        # Close first: a pump parked in the stream iterator is unblocked by the
        # channel erroring, not by the flag, so joining first would wait out the
        # full timeout for every live relay.
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._stub = None
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=5)

    def handle_request(self, relay_token: str, endpoint_id: str) -> None:
        """Serve one relay request without blocking the caller's stream."""
        if self._closing.is_set():
            return
        thread = threading.Thread(
            target=self._serve,
            args=(relay_token, endpoint_id),
            name=f"flowmesh-relay-{endpoint_id}",
            daemon=True,
        )
        with self._lock:
            self._threads.add(thread)
        thread.start()

    def _serve(self, relay_token: str, endpoint_id: str) -> None:
        try:
            port = self._endpoints.resolve(endpoint_id)
            if port is None:
                logger.warning("Refusing relay for unknown endpoint %s", endpoint_id)
                self._refuse(relay_token, endpoint_id)
                return
            self._pump(relay_token, endpoint_id, port)
        except grpc.RpcError as exc:
            logger.warning("Relay stream for %s failed: %s", endpoint_id, exc)
        except ValueError as exc:
            # Invoking on a channel shutdown() just closed raises this, not
            # RpcError. Scoped to shutdown so a genuine ValueError still surfaces.
            if not self._closing.is_set():
                raise
            logger.info("Relay for %s abandoned during shutdown: %s", endpoint_id, exc)
        except OSError as exc:
            logger.warning(
                "Relay for %s could not reach its port: %s", endpoint_id, exc
            )
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _open_frame(
        self, relay_token: str, endpoint_id: str
    ) -> supervisor_pb2.RelayFrame:
        return supervisor_pb2.RelayFrame(
            open=supervisor_pb2.RelayOpen(
                relay_token=relay_token, endpoint_id=endpoint_id
            )
        )

    def _refuse(self, relay_token: str, endpoint_id: str) -> None:
        """Open the stream only to close it, so the supervisor stops waiting."""
        stub = self._stub
        if stub is None:
            return
        frames = iter(
            [
                self._open_frame(relay_token, endpoint_id),
                supervisor_pb2.RelayFrame(eof=True),
            ]
        )
        responses = stub.Relay(frames, metadata=self._client.grpc_metadata())
        try:
            for _ in responses:
                pass
        except grpc.RpcError:
            pass

    def _pump(self, relay_token: str, endpoint_id: str, port: int) -> None:
        stub = self._stub
        if stub is None:
            return
        sock = socket.create_connection(("127.0.0.1", port), _CONNECT_TIMEOUT_SEC)
        sock.settimeout(None)
        finished = threading.Event()
        try:
            responses = stub.Relay(
                self._requests(sock, relay_token, endpoint_id, finished),
                metadata=self._client.grpc_metadata(),
            )
            for frame in responses:
                if frame.HasField("eof"):
                    break
                if frame.HasField("data"):
                    sock.sendall(frame.data)
        finally:
            finished.set()
            # Unblock the request generator, which is parked in recv().
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _requests(
        self,
        sock: socket.socket,
        relay_token: str,
        endpoint_id: str,
        finished: threading.Event,
    ) -> Iterator[supervisor_pb2.RelayFrame]:
        yield self._open_frame(relay_token, endpoint_id)
        while not finished.is_set() and not self._closing.is_set():
            try:
                chunk = sock.recv(_READ_CHUNK)
            except OSError:
                break
            if not chunk:
                break
            yield supervisor_pb2.RelayFrame(data=chunk)
        yield supervisor_pb2.RelayFrame(eof=True)
