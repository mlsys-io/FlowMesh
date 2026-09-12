"""The worker end of a relay: what it connects to, and how it tears down."""

import queue
import socket
import threading
import time

import grpc
import pytest

from shared.grpc.supervisor.v1 import supervisor_pb2
from worker.relay import EndpointRegistry, RelayClient


class _FakeStub:
    """Stands in for the generated stub, driving both halves of the stream."""

    def __init__(self) -> None:
        self.sent: list[supervisor_pb2.RelayFrame] = []
        self.opened = threading.Event()
        self.request_done = threading.Event()
        self._to_worker: queue.Queue[supervisor_pb2.RelayFrame | None] = queue.Queue()
        self._drain: threading.Thread | None = None

    def Relay(self, request_iterator, metadata=None):  # noqa: N802
        def drain() -> None:
            for frame in request_iterator:
                self.sent.append(frame)
                if frame.HasField("open"):
                    self.opened.set()
            self.request_done.set()

        self._drain = threading.Thread(target=drain, daemon=True)
        self._drain.start()
        return self._responses()

    def _responses(self):
        while True:
            frame = self._to_worker.get()
            if frame is None:
                return
            yield frame

    def push(self, frame: supervisor_pb2.RelayFrame) -> None:
        self._to_worker.put(frame)

    def close(self) -> None:
        self._to_worker.put(None)

    def data_sent(self) -> bytes:
        return b"".join(f.data for f in self.sent if f.HasField("data"))

    def saw_eof(self) -> bool:
        return any(f.HasField("eof") and f.eof for f in self.sent)


class _EchoServer:
    """A local listener standing in for a session's sshd."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self.received = bytearray()
        self.conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self._sock.accept()
        self.conn = conn
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            self.received.extend(chunk)

    def close_conn(self) -> None:
        """Send FIN. A bare close() would not, with _serve blocked in recv."""
        assert self.conn is not None
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class _FakeClient:
    def grpc_metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", "Bearer test"),)


def _make_client(stub: _FakeStub) -> tuple[RelayClient, EndpointRegistry]:
    endpoints = EndpointRegistry()
    relay = RelayClient(_FakeClient(), endpoints)  # type: ignore[arg-type]
    relay._stub = stub  # type: ignore[assignment]
    return relay, endpoints


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestRefusal:
    def test_unknown_endpoint_is_refused_without_connecting(self) -> None:
        """An id nobody published must not become a connection to anything."""
        stub = _FakeStub()
        relay, _ = _make_client(stub)
        stub.close()

        relay.handle_request("tok", "ssn-never-published")

        assert _wait(lambda: stub.request_done.is_set())
        assert stub.opened.is_set()
        assert stub.saw_eof()
        assert stub.data_sent() == b""

    def test_withdrawn_endpoint_is_refused(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)
        endpoints.withdraw("ssn-a")
        stub.close()

        relay.handle_request("tok", "ssn-a")

        assert _wait(lambda: stub.request_done.is_set())
        assert stub.data_sent() == b""
        assert server.conn is None
        server.close()


class TestPumping:
    def test_stream_data_reaches_the_local_port(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok", "ssn-a")
        assert _wait(lambda: stub.opened.is_set())
        stub.push(supervisor_pb2.RelayFrame(data=b"hello sshd"))

        assert _wait(lambda: bytes(server.received) == b"hello sshd")
        stub.close()
        server.close()

    def test_local_port_data_reaches_the_stream(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok", "ssn-a")
        assert _wait(lambda: server.conn is not None)
        assert server.conn is not None
        server.conn.sendall(b"SSH-2.0-OpenSSH")

        assert _wait(lambda: stub.data_sent() == b"SSH-2.0-OpenSSH")
        stub.close()
        server.close()

    def test_open_frame_names_the_token_and_endpoint(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok-123", "ssn-a")
        assert _wait(lambda: stub.opened.is_set())

        first = stub.sent[0]
        assert first.open.relay_token == "tok-123"
        assert first.open.endpoint_id == "ssn-a"
        stub.close()
        server.close()


class TestTeardown:
    def test_local_close_ends_the_stream(self) -> None:
        """The session going away must eof the relay, not hang it."""
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok", "ssn-a")
        assert _wait(lambda: server.conn is not None)
        server.close_conn()

        assert _wait(lambda: stub.saw_eof())
        stub.close()
        server.close()

    def test_stream_eof_closes_the_local_connection(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok", "ssn-a")
        assert _wait(lambda: server.conn is not None)
        stub.push(supervisor_pb2.RelayFrame(eof=True))

        assert _wait(lambda: server.conn is not None and server.conn.recv(1) == b"")
        server.close()

    def test_both_relay_threads_exit(self) -> None:
        """Sync gRPC adds a request-draining thread on top of our pump."""
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)

        relay.handle_request("tok", "ssn-a")
        assert _wait(lambda: server.conn is not None)
        stub.push(supervisor_pb2.RelayFrame(eof=True))
        stub.close()

        assert _wait(lambda: stub.request_done.is_set())
        assert _wait(
            lambda: not any(
                t.name.startswith("flowmesh-relay-") for t in threading.enumerate()
            )
        )
        server.close()


class TestDedicatedChannel:
    def test_relay_channel_is_a_second_connection(self) -> None:
        """gRPC pools subchannels by target and options, so identical channels
        would share one TCP connection and one flow-control window."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        target = f"127.0.0.1:{listener.getsockname()[1]}"
        accepted: list[socket.socket] = []
        stop = threading.Event()

        def accept_loop() -> None:
            listener.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = listener.accept()
                except (TimeoutError, OSError):
                    continue
                accepted.append(conn)

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()

        options = [("grpc.max_receive_message_length", 1024)]
        control = grpc.insecure_channel(target, options=options)
        shared = grpc.insecure_channel(target, options=options)
        dedicated = grpc.insecure_channel(
            target, options=[*options, ("grpc.use_local_subchannel_pool", 1)]
        )
        try:
            for channel in (control, shared, dedicated):
                channel.subscribe(lambda _state: None, try_to_connect=True)
            assert _wait(lambda: len(accepted) >= 2)
            time.sleep(0.5)
            # control and shared pool into one; dedicated opens its own.
            assert len(accepted) == 2
        finally:
            stop.set()
            thread.join(timeout=2)
            for channel in (control, shared, dedicated):
                channel.close()
            for conn in accepted:
                conn.close()
            listener.close()


class TestShutdown:
    def test_shutdown_closes_the_relay_channel(self) -> None:
        closed = threading.Event()

        class _Channel:
            def close(self) -> None:
                closed.set()

        stub = _FakeStub()
        relay, _ = _make_client(stub)
        relay._channel = _Channel()  # type: ignore[assignment]
        relay.shutdown()

        assert closed.is_set()

    def test_requests_after_shutdown_are_ignored(self) -> None:
        stub = _FakeStub()
        relay, endpoints = _make_client(stub)
        server = _EchoServer()
        endpoints.publish("ssn-a", server.port)
        relay.shutdown()

        relay.handle_request("tok", "ssn-a")

        assert not _wait(lambda: stub.opened.is_set(), timeout=0.5)
        server.close()


@pytest.fixture(autouse=True)
def _no_leaked_relay_threads():
    yield
    _wait(
        lambda: not any(
            t.name.startswith("flowmesh-relay-") for t in threading.enumerate()
        ),
        timeout=2.0,
    )
