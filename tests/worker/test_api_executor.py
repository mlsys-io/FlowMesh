"""Tests for the API executor url override and Nebula credential handling."""

import email.utils
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from worker.executors.api_executor import (
    _MAX_RETRIES,
    _RETRY_BACKOFF_MAX_SEC,
    APIExecutor,
)
from worker.executors.base_executor import ExecutionError, TaskCancelledError


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api",
        "workflow_id": "wf-1",
        "owner_id": "owner",
        "assigned_worker": "worker-1",
        "dispatched_at": "2026-03-22T00:00:00Z",
        "task": {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "wf:api"},
            "spec": {
                "taskType": "api",
                "api": {
                    "method": "POST",
                    "body": {"messages": [{"role": "user", "content": "hi"}]},
                    **spec_updates,
                },
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


class _RecordingTransport(httpx.MockTransport):
    """MockTransport that records the request it served."""

    def __init__(self) -> None:
        self.request: httpx.Request | None = None
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"total_tokens": 3},
            },
        )


class _BlockingTransport(httpx.MockTransport):
    """MockTransport that blocks on the first request, then serves a sequence."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            if not self.release.wait(5.0):
                raise AssertionError("blocking transport was not released")
        return self.responses.pop(0)


class _NotifyTransport(httpx.MockTransport):
    """MockTransport that sets an event once it has served a request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.served = threading.Event()
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.served.set()
        return self.responses.pop(0)


def _run(
    executor: APIExecutor, task: WorkerTaskMessage, transport: httpx.MockTransport
) -> None:
    executor._cancel_event = threading.Event()
    executor._cancel_lock = threading.Lock()
    executor._active_task_id = None
    executor._pending_cancelled_ids = set()
    with patch.object(
        APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
    ):
        executor.run(task, Path("/tmp/out"))


class TestNebulaPath:
    def test_no_url_no_header_uses_nebula_url_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer nebula-token"

    def test_no_url_with_header_preserves_header_and_skips_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers={"Authorization": "Bearer custom"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    def test_neither_url_nor_base_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_BASE_URL", raising=False)
        task = _task_message()
        with pytest.raises(ExecutionError, match="spec.api.url or NEBULA_API_BASE_URL"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())


class TestCustomUrl:
    def test_custom_url_without_credential_sends_no_nebula_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom endpoint may be unauthenticated, but never gets the Nebula token.

        An unauthorized serving endpoint must stay usable, so a missing
        credential is not an error. The Nebula token IS set in the environment
        here, so the absent Authorization header proves it is withheld rather
        than merely unavailable: a caller-chosen endpoint never receives it.
        """
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(url="https://custom.example.com/v1/chat/completions")
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert "Authorization" not in transport.request.headers

    def test_custom_url_with_header_preserves_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer custom"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    def test_custom_url_with_x_api_key_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom endpoint may authenticate with a non-Authorization header."""
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"X-API-Key": "custom-key"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://custom.example.com/v1/chat/completions"
        assert transport.request.headers["X-API-Key"] == "custom-key"

    def test_custom_url_with_only_innocent_header_stays_unauthenticated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-credential header leaves the request unauthenticated, not rejected.

        Content-Type is not a credential, so nothing here authenticates the
        request -- and the Nebula token still must not be substituted in.
        """
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(
            url="https://custom.example.com/v1/chat/completions",
            headers={"Content-Type": "application/json"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Content-Type"] == "application/json"
        assert "Authorization" not in transport.request.headers


class _SequenceTransport(httpx.MockTransport):
    """MockTransport that serves a fixed sequence of responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses.pop(0)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 3},
        },
    )


def _error_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, json={"error": "boom"})


class TestRetries:
    def _task(self, **spec_updates: object) -> WorkerTaskMessage:
        return _task_message(
            url="https://custom.example.com/v1/chat/completions",
            response={"parse_json": False},
            **spec_updates,
        )

    def test_retry_succeeds_after_transient_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 504 followed by a 200 succeeds when retries are configured."""
        monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _ok_response()]
        )
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.calls == 3

    def test_retries_exhausted_still_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistent 5xx failures exhaust retries and raise loudly."""
        monkeypatch.setattr("worker.executors.api_executor._RETRY_BACKOFF_SEC", 0.0)
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _error_response(504)]
        )
        with pytest.raises(ExecutionError, match="status 504"):
            _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.calls == 3

    def test_no_retry_by_default(self) -> None:
        """Without a retries field, a transient failure fails immediately."""
        task = self._task()
        transport = _SequenceTransport([_error_response(504)])
        with pytest.raises(ExecutionError, match="status 504"):
            _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.calls == 1

    def test_non_retryable_status_not_retried(self) -> None:
        """A 4xx (other than 408/429) is never retried."""
        task = self._task(retries=3)
        transport = _SequenceTransport([_error_response(400)])
        with pytest.raises(ExecutionError, match="status 400"):
            _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.calls == 1

    def test_cancelled_task_stops_retrying(self) -> None:
        """A cancelled task does not keep retrying."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()
        task = self._task(retries=3)
        executor.cancel(task.task_id)
        transport = _SequenceTransport([_error_response(504)])
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task, Path("/tmp/out"))
        assert transport.calls == 0

    def test_invalid_retries_rejected(self) -> None:
        """A negative or non-integer retries value is rejected."""
        for bad in (-1, "2", 1.5, True):
            task = self._task(retries=bad)
            with pytest.raises(ExecutionError, match="spec.api.retries"):
                _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_cancel_previous_task_does_not_cancel_next(self) -> None:
        """A cancellation left over from a prior task does not cancel the next."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()

        task_a = self._task(retries=0)
        task_a.task_id = "task-a"
        executor.cancel(task_a.task_id)

        task_b = self._task(retries=0)
        task_b.task_id = "task-b"
        transport = _RecordingTransport()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            executor.run(task_b, Path("/tmp/out"))
        assert transport.request is not None

    def test_late_cancel_of_previous_task_does_not_overwrite_next(self) -> None:
        """A late cancel for A cannot overwrite a recorded cancel for B."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()

        task_b = self._task(retries=0)
        task_b.task_id = "task-b"
        executor.cancel(task_b.task_id)
        executor.cancel("task-a")

        transport = _RecordingTransport()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task_b, Path("/tmp/out"))
        assert transport.request is None

    def test_delayed_cancel_of_previous_task_does_not_cancel_next(self) -> None:
        """A late cancellation for a prior task does not cancel a running task."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()

        task_b = self._task(retries=1)
        task_b.task_id = "task-b"
        # First request blocks; once released it returns a retryable 503 so the
        # loop re-checks the cancel event, then a 200 succeeds.
        transport = _BlockingTransport([_error_response(503), _ok_response()])

        errors: list[BaseException] = []

        def _run_b() -> None:
            try:
                with patch.object(
                    APIExecutor,
                    "_get_client",
                    return_value=httpx.Client(transport=transport),
                ):
                    executor.run(task_b, Path("/tmp/out"))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_run_b)
        thread.start()
        try:
            assert transport.started.wait(2.0)
            executor.cancel("task-a")
        finally:
            transport.release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert errors == []
        assert transport.calls == 2

    def test_cancel_of_active_task_still_cancels(self) -> None:
        """A cancellation addressed to the running task still cancels it."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()

        task_b = self._task(retries=3)
        task_b.task_id = "task-b"
        transport = _BlockingTransport([_error_response(503)])

        errors: list[BaseException] = []

        def _run_b() -> None:
            try:
                with patch.object(
                    APIExecutor,
                    "_get_client",
                    return_value=httpx.Client(transport=transport),
                ):
                    executor.run(task_b, Path("/tmp/out"))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_run_b)
        thread.start()
        try:
            assert transport.started.wait(2.0)
            executor.cancel("task-b")
        finally:
            transport.release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)

    def test_cancel_during_backoff_stops_retrying(self) -> None:
        """A cancellation during the retry backoff aborts well before it ends."""
        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()

        task = self._task(retries=3)
        task.task_id = "task-b"
        transport = _NotifyTransport([_error_response(503)])

        def _cancel_on_served() -> None:
            transport.served.wait(2.0)
            executor.cancel(task.task_id)

        canceller = threading.Thread(target=_cancel_on_served)
        canceller.start()
        start = time.monotonic()
        with patch.object(
            APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
        ):
            with pytest.raises(TaskCancelledError):
                executor.run(task, Path("/tmp/out"))
        elapsed = time.monotonic() - start
        canceller.join()
        assert elapsed < 0.5
        assert transport.calls == 1

    def _run_recording_delays(
        self, task: WorkerTaskMessage, transport: httpx.MockTransport
    ) -> list[float]:
        """Run a task, recording each backoff delay instead of waiting."""
        delays: list[float] = []

        def _record(delay: float) -> None:
            delays.append(delay)

        executor = APIExecutor.__new__(APIExecutor)
        executor._cancel_event = threading.Event()
        executor._cancel_lock = threading.Lock()
        executor._active_task_id = None
        executor._pending_cancelled_ids = set()
        with patch.object(APIExecutor, "_wait_for_backoff", side_effect=_record):
            with patch.object(
                APIExecutor,
                "_get_client",
                return_value=httpx.Client(transport=transport),
            ):
                executor.run(task, Path("/tmp/out"))
        return delays

    def test_retry_after_seconds_is_honoured(self) -> None:
        """A Retry-After in seconds sets the wait for the next attempt."""
        task = self._task(retries=1)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "5"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [5.0]

    def test_retry_after_date_is_honoured(self) -> None:
        """A Retry-After HTTP date sets the wait for the next attempt."""
        task = self._task(retries=1)
        retry_at = datetime.now(UTC) + timedelta(seconds=5)
        transport = _SequenceTransport(
            [
                httpx.Response(
                    429, headers={"Retry-After": email.utils.format_datetime(retry_at)}
                ),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [pytest.approx(5.0, abs=1.0)]

    def test_invalid_retry_after_falls_back_to_exponential(self) -> None:
        """A broken Retry-After falls back to the exponential schedule."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "nan"}),
                httpx.Response(429, headers={"Retry-After": "not-a-date"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [1.0, 2.0]

    def test_exponential_backoff_schedule(self) -> None:
        """Retries back off exponentially from the base delay."""
        task = self._task(retries=3)
        transport = _SequenceTransport(
            [
                _error_response(503),
                _error_response(503),
                _error_response(503),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [1.0, 2.0, 4.0]

    def test_retry_after_is_capped(self) -> None:
        """A hostile Retry-After cannot stall a worker past the cap."""
        task = self._task(retries=1)
        transport = _SequenceTransport(
            [
                httpx.Response(429, headers={"Retry-After": "999999"}),
                _ok_response(),
            ]
        )
        delays = self._run_recording_delays(task, transport)
        assert delays == [_RETRY_BACKOFF_MAX_SEC]

    def test_retries_above_maximum_rejected(self) -> None:
        """A retries value above the maximum is rejected."""
        task = self._task(retries=_MAX_RETRIES + 1)
        with pytest.raises(ExecutionError, match=f"at most {_MAX_RETRIES}"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_one_warning_per_retry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each retry logs one warning naming the attempt and the delay."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _ok_response()]
        )
        with caplog.at_level(logging.WARNING, logger="worker.executors.api_executor"):
            self._run_recording_delays(task, transport)
        warnings = [
            r for r in caplog.records if r.name == "worker.executors.api_executor"
        ]
        assert len(warnings) == 2
        assert "attempt 1/2" in warnings[0].getMessage()
        assert "attempt 2/2" in warnings[1].getMessage()
