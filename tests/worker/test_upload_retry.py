"""Tests for worker upload retries."""

import ssl
from typing import Any

import httpx
import pytest
import requests
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import MaxRetryError, NewConnectionError

from worker.utils import upload_retry
from worker.utils.upload_retry import send_with_retries


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}


def _sequence(*outcomes: _Response | Exception):
    remaining = list(outcomes)
    calls: list[int] = []

    def send() -> _Response:
        calls.append(len(calls))
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return send, calls


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    monkeypatch.delenv("WORKER_UPLOAD_RETRIES", raising=False)
    monkeypatch.delenv("WORKER_UPLOAD_BACKOFF_SEC", raising=False)
    fake = _Clock()
    monkeypatch.setattr(upload_retry, "time", fake)
    return fake


def _send(send: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("url", "http://server/api/v1/results")
    kwargs.setdefault("idempotent", True)
    return send_with_retries(send, what="upload", **kwargs)


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("Connection refused"),
        requests.ReadTimeout("read timeout=30.0"),
        httpx.ConnectError("Connection refused"),
        httpx.ReadTimeout("timed out"),
    ],
)
def test_connection_failures_are_retried_with_doubling_backoff(
    error: Exception, clock: _Clock
) -> None:
    send, calls = _sequence(error, error, _Response(200))

    response = _send(send)

    assert response.status_code == 200
    assert len(calls) == 3
    assert clock.delays == [2.0, 4.0]


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_transient_statuses_are_retried(status: int) -> None:
    send, calls = _sequence(_Response(status), _Response(200))

    response = _send(send)

    assert response.status_code == 200
    assert len(calls) == 2


@pytest.mark.parametrize(("header", "delay"), [("7", 7.0), ("35", 35.0), ("300", 60.0)])
def test_retry_after_overrides_backoff_under_the_cap(
    header: str, delay: float, clock: _Clock
) -> None:
    send, _calls = _sequence(_Response(503, {"Retry-After": header}), _Response(200))

    response = _send(send)

    assert response.status_code == 200
    assert clock.delays == [delay]


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 413])
def test_final_and_client_error_statuses_are_returned_at_once(status: int) -> None:
    send, calls = _sequence(_Response(status))

    response = _send(send)

    assert response.status_code == status
    assert len(calls) == 1


def test_last_transient_status_is_returned_to_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "1")
    send, calls = _sequence(_Response(503), _Response(503))

    response = _send(send)

    assert response.status_code == 503
    assert len(calls) == 2


def test_last_connection_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "2")
    error = requests.ConnectionError("Connection refused")
    send, calls = _sequence(error, error, error)

    with pytest.raises(requests.ConnectionError):
        _send(send)
    assert len(calls) == 3


def test_other_errors_are_not_retried() -> None:
    send, calls = _sequence(ValueError("bad payload"))

    with pytest.raises(ValueError):
        _send(send)
    assert len(calls) == 1


def test_zero_retries_sends_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "0")
    send, calls = _sequence(requests.ConnectionError("refused"))

    with pytest.raises(requests.ConnectionError):
        _send(send)
    assert len(calls) == 1


def test_backoff_is_capped(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "6")
    error = requests.ConnectionError("refused")
    send, _calls = _sequence(*([error] * 6), _Response(200))

    _send(send)

    assert clock.delays == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0]


def test_no_retry_follows_an_attempt_that_used_up_the_deadline(
    clock: _Clock,
) -> None:
    calls = 0

    def send() -> _Response:
        nonlocal calls
        calls += 1
        clock.now += 1800
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(httpx.ReadTimeout):
        _send(send, deadline_sec=1800)
    assert calls == 1
    assert clock.delays == []


def test_retries_stop_when_the_next_wait_would_pass_the_deadline(
    clock: _Clock,
) -> None:
    send, calls = _sequence(*([_Response(503)] * 6))

    response = _send(send, deadline_sec=10)

    assert response.status_code == 503
    assert len(calls) == 3
    assert clock.delays == [2.0, 4.0]


def _certificate_failure(error: Exception) -> Exception:
    error.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed")
    return error


@pytest.mark.parametrize(
    "error",
    [
        _certificate_failure(requests.exceptions.SSLError("verify failed")),
        _certificate_failure(httpx.ConnectError("verify failed")),
        httpx.UnsupportedProtocol("Request URL is missing a scheme"),
        httpx.LocalProtocolError("Illegal header value"),
    ],
)
def test_permanent_transport_errors_are_not_retried(error: Exception) -> None:
    send, calls = _sequence(error, _Response(200))

    with pytest.raises(type(error)):
        _send(send)
    assert len(calls) == 1


def _refused() -> requests.ConnectionError:
    reason = NewConnectionError(
        HTTPConnection("sink"), "[Errno 111] Connection refused"
    )
    return requests.ConnectionError(
        MaxRetryError(HTTPConnectionPool("sink"), "/x", reason)
    )


@pytest.mark.parametrize(
    "error",
    [
        _refused(),
        requests.ConnectTimeout("connect timeout"),
        httpx.ConnectError("refused"),
    ],
)
def test_non_idempotent_upload_retries_a_request_that_never_left(
    error: Exception,
) -> None:
    send, calls = _sequence(error, _Response(200))

    response = _send(send, idempotent=False)

    assert response.status_code == 200
    assert len(calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        requests.ReadTimeout("read timeout=30.0"),
        requests.ConnectionError("Connection aborted"),
        httpx.ReadTimeout("timed out"),
    ],
)
def test_non_idempotent_upload_is_not_resent_after_it_may_have_landed(
    error: Exception,
) -> None:
    send, calls = _sequence(error)

    with pytest.raises(type(error)):
        _send(send, idempotent=False)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_non_idempotent_upload_does_not_retry_http_statuses(status: int) -> None:
    send, calls = _sequence(_Response(status), _Response(200))

    response = _send(send, idempotent=False)

    assert response.status_code == status
    assert len(calls) == 1


def test_tls_errors_other_than_certificate_verification_are_retried() -> None:
    error = httpx.ConnectError("EOF occurred in violation of protocol")
    error.__cause__ = ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
    send, calls = _sequence(error, _Response(200))

    assert _send(send).status_code == 200
    assert len(calls) == 2


def test_an_unreachable_origin_gets_one_attempt_until_it_answers(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "2")
    error = requests.ConnectionError("Connection refused")
    first, first_calls = _sequence(error, error, error)
    with pytest.raises(requests.ConnectionError):
        _send(first, url="http://server/api/v1/results/tsk-1/files")

    second, second_calls = _sequence(error)
    with pytest.raises(requests.ConnectionError):
        _send(second, url="http://SERVER/api/v1/traces/tasks/tsk-1/spans")

    other, other_calls = _sequence(error, _Response(200))
    assert _send(other, url="http://other/sink").status_code == 200

    answered, _ = _sequence(_Response(404))
    assert _send(answered).status_code == 404
    recovered, recovered_calls = _sequence(error, _Response(200))
    assert _send(recovered).status_code == 200

    assert (len(first_calls), len(second_calls)) == (3, 1)
    assert (len(other_calls), len(recovered_calls)) == (2, 2)
    assert clock.delays == [2.0, 4.0, 2.0, 2.0]


def test_a_failure_that_may_have_landed_does_not_mark_the_origin() -> None:
    send, _calls = _sequence(requests.ReadTimeout("read timeout=30.0"))
    with pytest.raises(requests.ReadTimeout):
        _send(send, url="http://external/sink", idempotent=False)

    retry, calls = _sequence(requests.ConnectTimeout("connect timeout"), _Response(200))
    assert _send(retry, url="http://external/sink", idempotent=False).status_code == 200
    assert len(calls) == 2
