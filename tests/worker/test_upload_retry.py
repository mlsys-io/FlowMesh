"""Tests for worker upload retries."""

import httpx
import pytest
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

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


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_UPLOAD_RETRIES", raising=False)
    monkeypatch.delenv("WORKER_UPLOAD_BACKOFF_SEC", raising=False)


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
    error: Exception,
) -> None:
    send, calls = _sequence(error, error, _Response(200))
    delays: list[float] = []

    response = send_with_retries(send, what="upload", sleep=delays.append)

    assert response.status_code == 200
    assert len(calls) == 3
    assert delays == [2.0, 4.0]


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_transient_statuses_are_retried(status: int) -> None:
    send, calls = _sequence(_Response(status), _Response(200))

    response = send_with_retries(send, what="upload", sleep=lambda _s: None)

    assert response.status_code == 200
    assert len(calls) == 2


def test_retry_after_overrides_backoff() -> None:
    send, _calls = _sequence(_Response(503, {"Retry-After": "7"}), _Response(200))
    delays: list[float] = []

    send_with_retries(send, what="upload", sleep=delays.append)

    assert delays == [7.0]


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 413])
def test_final_and_client_error_statuses_are_returned_at_once(status: int) -> None:
    send, calls = _sequence(_Response(status))

    response = send_with_retries(send, what="upload", sleep=lambda _s: None)

    assert response.status_code == status
    assert len(calls) == 1


def test_last_transient_status_is_returned_to_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "1")
    send, calls = _sequence(_Response(503), _Response(503))

    response = send_with_retries(send, what="upload", sleep=lambda _s: None)

    assert response.status_code == 503
    assert len(calls) == 2


def test_last_connection_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "2")
    error = requests.ConnectionError("Connection refused")
    send, calls = _sequence(error, error, error)

    with pytest.raises(requests.ConnectionError):
        send_with_retries(send, what="upload", sleep=lambda _s: None)
    assert len(calls) == 3


def test_other_errors_are_not_retried() -> None:
    send, calls = _sequence(ValueError("bad payload"))

    with pytest.raises(ValueError):
        send_with_retries(send, what="upload", sleep=lambda _s: None)
    assert len(calls) == 1


def test_zero_retries_sends_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "0")
    send, calls = _sequence(requests.ConnectionError("refused"))

    with pytest.raises(requests.ConnectionError):
        send_with_retries(send, what="upload", sleep=lambda _s: None)
    assert len(calls) == 1


def test_backoff_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RETRIES", "6")
    error = requests.ConnectionError("refused")
    send, _calls = _sequence(*([error] * 6), _Response(200))
    delays: list[float] = []

    send_with_retries(send, what="upload", sleep=delays.append)

    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def _refused() -> requests.ConnectionError:
    reason = NewConnectionError(None, "[Errno 111] Connection refused")  # type: ignore[arg-type]
    return requests.ConnectionError(MaxRetryError(None, "/x", reason))  # type: ignore[arg-type]


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

    response = send_with_retries(
        send, what="upload", idempotent=False, sleep=lambda _s: None
    )

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
        send_with_retries(send, what="upload", idempotent=False, sleep=lambda _s: None)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "retried"), [(429, True), (503, True), (500, False), (502, False)]
)
def test_non_idempotent_upload_retries_only_unprocessed_statuses(
    status: int, retried: bool
) -> None:
    send, calls = _sequence(_Response(status), _Response(200))

    response = send_with_retries(
        send, what="upload", idempotent=False, sleep=lambda _s: None
    )

    assert response.status_code == (200 if retried else status)
    assert len(calls) == (2 if retried else 1)
