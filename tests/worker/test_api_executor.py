"""Tests for the API executor's url override, Nebula credential handling, and
batch mode (one task, N row-aligned requests)."""

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from shared.schemas.result import APIGroupItem, APIItem, APIResult
from shared.tasks.worker_message import WorkerTaskMessage
from worker.executors import api_executor as api_executor_module
from worker.executors.api_executor import APIExecutor
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
                "data": {"type": "list", "items": ["hi"]},
                "api": {
                    "method": "POST",
                    "body": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
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


class _EchoTransport(httpx.MockTransport):
    """MockTransport that echoes each row's prompt back with row-specific
    status, usage, and headers."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = request.read()
        prompt = json.loads(body)["messages"][0]["content"]
        return httpx.Response(
            200 + len(prompt) % 3,
            headers={"X-Row": prompt},
            json={
                "choices": [{"message": {"content": f"echo:{prompt}"}}],
                "usage": {"total_tokens": len(prompt)},
            },
        )


def _run(
    executor: APIExecutor,
    task: WorkerTaskMessage,
    transport: httpx.MockTransport,
    out_dir: Path = Path("/tmp/out"),
):
    with patch.object(
        APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
    ):
        return executor.run(task, out_dir)


def _executor() -> APIExecutor:
    """Build an APIExecutor with cancellation state, without a WorkerConfig."""
    executor = APIExecutor.__new__(APIExecutor)
    executor._cancel_event = threading.Event()
    executor._cancel_task_id = None
    executor._cancel_lock = threading.Lock()
    executor._task_id = None
    executor._current_batch_id = None
    return executor


def _batch_task(items: list[Any], **api_updates: Any) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api-batch",
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
                    "body": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
                    "response": {
                        "raise_for_status": False,
                        "include_headers": True,
                    },
                    **api_updates,
                },
                "data": {"type": "list", "items": items},
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


def _api_item(content: str) -> APIItem:
    """An upstream APIItem whose response carries the given content."""
    item = APIItem(index=0, url="u", status_code=200)
    item.response_json = {"choices": [{"message": {"content": content}}]}
    return item


class TestNebulaPath:
    def test_no_url_no_header_uses_nebula_url_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(_executor(), task, transport)
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
        _run(_executor(), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"
        assert transport.request.headers["Authorization"] == "Bearer custom"

    def test_neither_url_nor_base_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_BASE_URL", raising=False)
        task = _task_message()
        with pytest.raises(ExecutionError, match="spec.api.url or NEBULA_API_BASE_URL"):
            _run(_executor(), task, _RecordingTransport())


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
        _run(_executor(), task, transport)
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
        _run(_executor(), task, transport)
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
        _run(_executor(), task, transport)
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
        _run(_executor(), task, transport)
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

    def test_retry_succeeds_after_transient_failures(self) -> None:
        """A 504 followed by a 200 succeeds when retries are configured."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _ok_response()]
        )
        _run(_executor(), task, transport)
        assert transport.calls == 3

    def test_retries_exhausted_still_fails(self) -> None:
        """Persistent 5xx failures exhaust retries and raise loudly."""
        task = self._task(retries=2)
        transport = _SequenceTransport(
            [_error_response(504), _error_response(504), _error_response(504)]
        )
        with pytest.raises(ExecutionError, match="status 504"):
            _run(_executor(), task, transport)
        assert transport.calls == 3

    def test_no_retry_by_default(self) -> None:
        """Without a retries field, a transient failure fails immediately."""
        task = self._task()
        transport = _SequenceTransport([_error_response(504)])
        with pytest.raises(ExecutionError, match="status 504"):
            _run(_executor(), task, transport)
        assert transport.calls == 1

    def test_non_retryable_status_not_retried(self) -> None:
        """A 4xx (other than 408/429) is never retried."""
        task = self._task(retries=3)
        transport = _SequenceTransport([_error_response(400)])
        with pytest.raises(ExecutionError, match="status 400"):
            _run(_executor(), task, transport)
        assert transport.calls == 1

    def test_cancelled_task_stops_retrying(self) -> None:
        """A cancelled task does not keep retrying."""
        executor = _executor()
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
                _run(_executor(), task, _RecordingTransport())


class TestBatch:
    @pytest.fixture(autouse=True)
    def _nebula_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")

    def test_issues_one_request_per_row_in_order(self, tmp_path: Path) -> None:
        task = _batch_task(["first", "second", "third"])
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 3
        issued = {
            json.loads(req.read())["messages"][0]["content"]
            for req in transport.requests
        }
        assert issued == {"first", "second", "third"}
        for idx, prompt in enumerate(["first", "second", "third"]):
            item = result.items[idx]
            assert item.index == idx
            assert item.prompt == prompt
            assert item.text == f"echo:{prompt}"
            assert item.response_json["choices"][0]["message"]["content"] == (
                f"echo:{prompt}"
            )
            assert item.status_code == 200 + len(prompt) % 3
            assert item.usage == {"total_tokens": len(prompt)}
            assert item.headers["x-row"] == prompt

    def test_rows_stay_aligned_when_requests_complete_out_of_order(
        self, tmp_path: Path
    ) -> None:
        """Output row i corresponds to input row i even when requests finish
        in reverse order."""

        class _ReverseTransport(httpx.MockTransport):
            def __init__(self) -> None:
                self.requests: list[httpx.Request] = []
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                prompt = json.loads(request.read())["messages"][0]["content"]
                delay = {"a": 0.3, "b": 0.2, "c": 0.1}[prompt]
                time.sleep(delay)
                return httpx.Response(
                    200 + len(prompt) % 3,
                    headers={"X-Row": prompt},
                    json={
                        "choices": [{"message": {"content": f"echo:{prompt}"}}],
                        "usage": {"total_tokens": len(prompt)},
                    },
                )

        task = _batch_task(["a", "b", "c"])
        transport = _ReverseTransport()
        result = _run(_executor(), task, transport, tmp_path)
        for idx, prompt in enumerate(["a", "b", "c"]):
            item = result.items[idx]
            assert item.index == idx
            assert item.prompt == prompt
            assert item.text == f"echo:{prompt}"
            assert item.response_json["choices"][0]["message"]["content"] == (
                f"echo:{prompt}"
            )
            assert item.status_code == 200 + len(prompt) % 3
            assert item.usage == {"total_tokens": len(prompt)}
            assert item.headers["x-row"] == prompt

    def test_single_row_batches_to_one_item(self, tmp_path: Path) -> None:
        task = _batch_task(["only"])
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 1
        assert len(result.items) == 1
        assert result.items[0].index == 0
        assert result.items[0].prompt == "only"

    def test_row_failure_fails_whole_task_without_shifting(
        self, tmp_path: Path
    ) -> None:
        class _FailRow(httpx.MockTransport):
            def __init__(self, failing_prompt: str) -> None:
                self.failing_prompt = failing_prompt
                self.requests: list[httpx.Request] = []
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                prompt = json.loads(request.read())["messages"][0]["content"]
                if prompt == self.failing_prompt:
                    return httpx.Response(
                        500,
                        json={
                            "choices": [{"message": {"content": "boom"}}],
                            "usage": {"total_tokens": 1},
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"total_tokens": 1},
                    },
                )

        task = _batch_task(["a", "b", "c"], response={"raise_for_status": True})
        transport = _FailRow("b")
        with pytest.raises(ExecutionError, match="row 1"):
            _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 3

    def test_placeholder_not_required_for_scalar_body(self, tmp_path: Path) -> None:
        """A batch task whose body has no placeholder still issues N requests."""
        payload = {
            "task_id": "task-api-batch",
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
                        "body": {"messages": [{"role": "user", "content": "static"}]},
                    },
                    "data": {"type": "list", "items": ["a", "b"]},
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 2
        assert len(result.items) == 2

    def test_no_rows_raises(self, tmp_path: Path) -> None:
        task = _batch_task([])
        with pytest.raises(ExecutionError, match="no rows"):
            _run(
                _executor(),
                task,
                _EchoTransport(),
                tmp_path,
            )

    def test_missing_data_raises(self, tmp_path: Path) -> None:
        """spec.data is required; an api task without it fails closed."""
        payload = {
            "task_id": "task-api-batch",
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
                    },
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        with pytest.raises(ExecutionError, match="spec.data is required"):
            _run(
                _executor(),
                task,
                _EchoTransport(),
                tmp_path,
            )

    def test_requests_issue_in_parallel(self, tmp_path: Path) -> None:
        """N rows take ~one row's latency, not N x, on a network-bound path."""

        class _SlowTransport(httpx.MockTransport):
            def __init__(self) -> None:
                self.requests: list[httpx.Request] = []
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                time.sleep(0.2)
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"total_tokens": 3},
                    },
                )

        n_rows = 4
        task = _batch_task([f"row-{i}" for i in range(n_rows)])
        transport = _SlowTransport()
        start = time.monotonic()
        result = _run(_executor(), task, transport, tmp_path)
        elapsed = time.monotonic() - start

        assert len(transport.requests) == n_rows
        assert elapsed < 0.2 * n_rows * 0.6
        assert [item.index for item in result.items] == list(range(n_rows))

    def test_concurrency_one_serializes_requests(self, tmp_path: Path) -> None:
        """concurrency: 1 limits the worker pool so requests never overlap."""

        class _OverlapTransport(httpx.MockTransport):
            def __init__(self) -> None:
                self.max_in_flight = 0
                self._in_flight = 0
                self._lock = threading.Lock()
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                with self._lock:
                    self._in_flight += 1
                    self.max_in_flight = max(self.max_in_flight, self._in_flight)
                time.sleep(0.05)
                with self._lock:
                    self._in_flight -= 1
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"total_tokens": 3},
                    },
                )

        task = _batch_task(["a", "b", "c", "d"], concurrency=1)
        transport = _OverlapTransport()
        _run(_executor(), task, transport, tmp_path)
        assert transport.max_in_flight == 1

    def test_request_skeleton_constructed_once(self, tmp_path: Path) -> None:
        """The request template is built once, not once per row."""
        task = _batch_task(["a", "b", "c"])
        transport = _EchoTransport()
        real_build = APIExecutor._build_request_kwargs

        with (
            patch.object(
                APIExecutor,
                "_get_client",
                return_value=httpx.Client(transport=transport),
            ),
            patch.object(
                APIExecutor, "_build_request_kwargs", autospec=True
            ) as mock_build,
        ):
            mock_build.side_effect = lambda *a, **k: real_build(*a, **k)
            _executor().run(task, tmp_path)

        assert mock_build.call_count == 1
        assert mock_build.call_args.args[2] is None

    @pytest.mark.parametrize("concurrency", [1, 4, 8])
    def test_client_pool_sized_to_concurrency(self, concurrency: int) -> None:
        """The connection pool matches the effective concurrency."""
        APIExecutor.close_all_clients()
        try:
            client = APIExecutor._get_client(
                "https://example.com",
                httpx.Timeout(60),
                True,
                True,
                concurrency,
            )
            pool = cast(Any, client._transport)._pool
            assert pool._max_connections == concurrency
            assert pool._max_keepalive_connections == concurrency
        finally:
            APIExecutor.close_all_clients()

    def test_concurrency_capped_at_max(self, tmp_path: Path) -> None:
        """A configured concurrency above the cap is clamped to the cap."""
        task = _batch_task(["a", "b", "c"], concurrency=100)
        captured: dict[str, Any] = {}

        def _fake_get_client(*args: Any, **kwargs: Any) -> httpx.Client:
            captured["concurrency"] = kwargs.get("concurrency", args[4])
            return httpx.Client(transport=_EchoTransport())

        with patch.object(APIExecutor, "_get_client", side_effect=_fake_get_client):
            _executor().run(task, tmp_path)

        assert captured["concurrency"] == 8

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_run_passes_effective_concurrency_to_client(
        self, tmp_path: Path, concurrency: int
    ) -> None:
        """run() forwards the uncapped configured concurrency to the client."""
        task = _batch_task(["a", "b", "c"], concurrency=concurrency)
        captured: dict[str, Any] = {}

        def _fake_get_client(*args: Any, **kwargs: Any) -> httpx.Client:
            captured["concurrency"] = kwargs.get("concurrency", args[4])
            return httpx.Client(transport=_EchoTransport())

        with patch.object(APIExecutor, "_get_client", side_effect=_fake_get_client):
            _executor().run(task, tmp_path)

        assert captured["concurrency"] == concurrency

    @pytest.mark.parametrize("concurrency", [0, -1])
    def test_concurrency_below_one_rejected(
        self, tmp_path: Path, concurrency: int
    ) -> None:
        """A configured concurrency below 1 is rejected."""
        task = _batch_task(["a", "b", "c"], concurrency=concurrency)
        with pytest.raises(ExecutionError, match="spec.api.concurrency must be >= 1"):
            _run(_executor(), task, _EchoTransport(), tmp_path)

    def test_client_cache_key_includes_concurrency(self) -> None:
        """Pools built for different concurrency values are not shared."""
        APIExecutor.close_all_clients()
        try:
            c1 = APIExecutor._get_client(
                "https://example.com", httpx.Timeout(60), True, True, 1
            )
            c4 = APIExecutor._get_client(
                "https://example.com", httpx.Timeout(60), True, True, 4
            )
            assert c1 is not c4
            assert len(APIExecutor._clients) == 2
        finally:
            APIExecutor.close_all_clients()

    def test_no_usage_2xx_produces_aligned_item(self, tmp_path: Path) -> None:
        """A 2xx response without usage still yields a row-aligned item."""

        class _NoUsage(httpx.MockTransport):
            def __init__(self) -> None:
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "ok"}}]},
                )

        task = _batch_task(["a", "b"])
        result = _run(_executor(), task, _NoUsage(), tmp_path)
        assert len(result.items) == 2
        assert result.items[0].text == "ok"
        assert result.items[0].usage is None

    def test_5xx_with_raise_for_status_false_produces_aligned_item(
        self, tmp_path: Path
    ) -> None:
        """A 5xx with raise_for_status false still yields a row-aligned item."""

        class _ErrorBody(httpx.MockTransport):
            def __init__(self) -> None:
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    503,
                    json={"error": {"message": "overloaded"}},
                )

        task = _batch_task(["a", "b"], response={"raise_for_status": False})
        result = _run(_executor(), task, _ErrorBody(), tmp_path)
        assert len(result.items) == 2
        assert result.items[0].status_code == 503
        assert result.items[0].response_json == {"error": {"message": "overloaded"}}

    def test_retryable_503_raises_retryable(self, tmp_path: Path) -> None:
        """A 503 is classified retryable even without a success payload."""

        class _ErrorBody(httpx.MockTransport):
            def __init__(self) -> None:
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    503,
                    json={"error": {"message": "overloaded"}},
                )

        task = _batch_task(["a", "b"], response={"raise_for_status": True})
        with pytest.raises(ExecutionError, match="status 503") as excinfo:
            _run(_executor(), task, _ErrorBody(), tmp_path)
        assert excinfo.value.retryable is True

    def test_cancel_prevents_queued_rows_from_issuing(self, tmp_path: Path) -> None:
        """After cancel, a row that has not started never issues its request."""

        class _BlockingTransport(httpx.MockTransport):
            def __init__(self) -> None:
                self.requests: list[httpx.Request] = []
                self.started = threading.Event()
                self.release = threading.Event()
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                self.started.set()
                self.release.wait(timeout=5)
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"total_tokens": 3},
                    },
                )

        executor = _executor()
        task = _batch_task(["a", "b", "c", "d"], concurrency=1)
        transport = _BlockingTransport()
        errors: list[BaseException] = []
        submitted: list[Any] = []
        all_submitted = threading.Event()
        real_submit = concurrent.futures.ThreadPoolExecutor.submit

        def _recording_submit(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
            submitted.append(fn)
            if len(submitted) == 4:
                all_submitted.set()
            return real_submit(self, fn, *args, **kwargs)

        def _run_in_thread() -> None:
            try:
                with patch.object(
                    concurrent.futures.ThreadPoolExecutor,
                    "submit",
                    _recording_submit,
                ):
                    _run(executor, task, transport, tmp_path)
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        assert transport.started.wait(timeout=5)
        assert all_submitted.wait(timeout=5)
        executor.cancel("task-api-batch")
        transport.release.set()
        thread.join(timeout=10)

        assert len(transport.requests) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)

    def test_cancel_before_submission_prevents_any_future(self, tmp_path: Path) -> None:
        """Cancelling before futures are submitted surfaces TaskCancelledError
        without submitting any future."""
        executor = _executor()
        task = _batch_task(["a", "b", "c", "d"])
        transport = _EchoTransport()
        errors: list[BaseException] = []
        submitted: list[Any] = []
        real_submit = concurrent.futures.ThreadPoolExecutor.submit
        real_base_url = APIExecutor._base_url

        def _recording_submit(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
            submitted.append(fn)
            return real_submit(self, fn, *args, **kwargs)

        def _run_in_thread() -> None:
            def _cancel_then_base_url(url: str) -> str:
                executor.cancel("task-api-batch")
                return real_base_url(url)

            try:
                with (
                    patch.object(
                        concurrent.futures.ThreadPoolExecutor,
                        "submit",
                        _recording_submit,
                    ),
                    patch.object(
                        APIExecutor,
                        "_base_url",
                        side_effect=_cancel_then_base_url,
                    ),
                ):
                    _run(executor, task, transport, tmp_path)
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join(timeout=10)

        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)
        assert submitted == []

    def test_cancel_after_requests_complete_before_collection_not_done(
        self, tmp_path: Path
    ) -> None:
        """A cancel arriving after every request has completed but before results
        are collected fails the task rather than returning DONE."""

        class _RecordingTransport(httpx.MockTransport):
            def __init__(self) -> None:
                self.requests: list[httpx.Request] = []
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"total_tokens": 3},
                    },
                )

        executor = _executor()
        task = _batch_task(["a", "b"], concurrency=2)
        transport = _RecordingTransport()
        errors: list[BaseException] = []
        futures: list[Any] = []
        collect_release = threading.Event()
        real_submit = concurrent.futures.ThreadPoolExecutor.submit
        real_as_completed = concurrent.futures.as_completed

        def _recording_submit(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
            future = real_submit(self, fn, *args, **kwargs)
            futures.append(future)
            return future

        def _blocking_as_completed(fs: Any, timeout: float | None = None) -> Any:
            collect_release.wait(timeout=5)
            return real_as_completed(fs, timeout=timeout)

        def _run_in_thread() -> None:
            try:
                with (
                    patch.object(
                        concurrent.futures.ThreadPoolExecutor,
                        "submit",
                        _recording_submit,
                    ),
                    patch.object(
                        api_executor_module, "as_completed", _blocking_as_completed
                    ),
                ):
                    _run(executor, task, transport, tmp_path)
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        assert transport.requests or True
        while len(futures) < 2:
            time.sleep(0.01)
        for future in futures:
            assert future.done()
        executor.cancel("task-api-batch")
        collect_release.set()
        thread.join(timeout=10)

        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)

    def test_cancel_before_run_cancels(self, tmp_path: Path) -> None:
        """A cancel that lands before run() starts still cancels the run."""
        executor = _executor()
        task = _batch_task(["a", "b"])
        transport = _EchoTransport()

        executor.cancel("task-api-batch")

        with pytest.raises(TaskCancelledError):
            _run(executor, task, transport, tmp_path)
        assert transport.requests == []

    def test_cancel_during_run_setup_not_lost(self, tmp_path: Path) -> None:
        """A cancel landing mid check-and-clear is not dropped."""
        executor = _executor()
        task = _batch_task(["a", "b"])
        transport = _EchoTransport()
        errors: list[BaseException] = []
        in_clear = threading.Event()
        release_clear = threading.Event()
        real_clear = executor._cancel_event.clear

        def _blocking_clear() -> None:
            in_clear.set()
            release_clear.wait(timeout=5)
            real_clear()

        executor._cancel_event.clear = _blocking_clear  # type: ignore[method-assign]

        def _run_in_thread() -> None:
            try:
                _run(executor, task, transport, tmp_path)
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        assert in_clear.wait(timeout=5)
        executor.cancel(task.task_id)
        release_clear.set()
        thread.join(timeout=10)

        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelledError)


class TestPromptSubstitution:
    def test_exact_placeholder_substitutes_raw_object(self) -> None:
        """A body value that is exactly {{prompt}} is replaced by the prompt
        object as-is, so a message list stays a list of dicts."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        payload = {
            "task_id": "task-api-sub",
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
                        "url": "https://custom.example.com/v1/chat/completions",
                        "json": {"messages": "{{prompt}}"},
                    },
                    "data": {"type": "list", "items": [messages]},
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        _run(_executor(), task, transport)
        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].read())
        assert body["messages"] == messages

    def test_embedded_placeholder_substitutes_string(self) -> None:
        """An embedded {{prompt}} inside a longer string keeps string
        substitution."""
        payload = {
            "task_id": "task-api-sub",
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
                        "url": "https://custom.example.com/v1/chat/completions",
                        "json": {
                            "messages": [{"role": "user", "content": "Q: {{prompt}}"}]
                        },
                    },
                    "data": {"type": "list", "items": ["hello"]},
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        _run(_executor(), task, transport)
        body = json.loads(transport.requests[0].read())
        assert body["messages"][0]["content"] == "Q: hello"


class TestDataframeRows:
    def test_dataframe_column_issues_one_request_per_row(self, tmp_path: Path) -> None:
        """A dataframe-spec API task whose column reads an upstream APIResult's
        items issues one request per upstream row, each body carrying that
        row's messages as a list."""
        upstream = APIResult(
            ok=True,
            executor="api",
            method="POST",
            url="https://up.example.com",
            status_code=200,
            items=[_api_item("c0"), _api_item("c1")],
        )
        payload = {
            "task_id": "task-api-df",
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
                    "_upstreamResults": {"Up": upstream},
                    "api": {
                        "method": "POST",
                        "url": "https://custom.example.com/v1/chat/completions",
                        "json": {"messages": "{{prompt}}"},
                    },
                    "data": {
                        "type": "dataframe",
                        "columns": [
                            {
                                "label": "L",
                                "node": "Up",
                                "path": "items.json.choices[0].message.content",
                            }
                        ],
                        "messages": [
                            {"role": "user", "content": "row {L}"},
                        ],
                    },
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 2
        issued = {
            json.loads(req.read())["messages"][0]["content"]
            for req in transport.requests
        }
        assert issued == {"row c0", "row c1"}
        # A single-column dataframe is one table, so one group item holds both rows.
        assert len(result.items) == 1
        prompts = [json.loads(r.prompt) for r in result.items[0].rows]
        assert prompts == [
            [{"role": "user", "content": "row c0"}],
            [{"role": "user", "content": "row c1"}],
        ]


class TestGraphTemplateAggregate:
    def test_graph_template_aggregates_all_rows_into_one_prompt(
        self, tmp_path: Path
    ) -> None:
        """A graph_template aggregate API task over all rows of an upstream
        APIResult issues one request whose prompt contains every row."""
        upstream = APIResult(
            ok=True,
            executor="api",
            method="POST",
            url="https://up.example.com",
            status_code=200,
            items=[_api_item("c0"), _api_item("c1")],
        )
        payload = {
            "task_id": "task-api-gt",
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
                    "_upstreamResults": {"Up": upstream},
                    "api": {
                        "method": "POST",
                        "url": "https://custom.example.com/v1/chat/completions",
                        "json": {"messages": "{{prompt}}"},
                    },
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "df",
                                    "data": {
                                        "type": "dataframe",
                                        "columns": [
                                            {
                                                "label": "L",
                                                "node": "Up",
                                                "path": (
                                                    "items.json.choices[0].message.content"
                                                ),
                                            }
                                        ],
                                    },
                                }
                            ],
                            "options": {
                                "format": {
                                    "steps": [],
                                    "messages": [
                                        {"role": "user", "content": "all: {df}"}
                                    ],
                                }
                            },
                        },
                    },
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].read())
        content = body["messages"][0]["content"]
        assert "c0" in content and "c1" in content
        assert len(result.items) == 1
        # Aggregate result is a plain item (read at items.json...), not a group
        # item (read at items.rows.json...).
        item = result.items[0]
        assert not hasattr(item, "rows")
        assert item.response_json["choices"][0]["message"]["content"].startswith(
            "echo:all:"
        )


class TestGroupedResult:
    def test_ragged_groups_return_one_item_per_group(self, tmp_path: Path) -> None:
        """A ragged grouped dataframe API task (groups of different sizes)
        returns one item per group with the right rows in each."""
        upstream = APIResult(
            ok=True,
            executor="api",
            method="POST",
            url="https://up.example.com",
            status_code=200,
            items=[
                APIGroupItem(index=0, rows=[_api_item("c0"), _api_item("c1")]),
                APIGroupItem(index=1, rows=[_api_item("c2")]),
            ],
        )
        payload = {
            "task_id": "task-api-grp",
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
                    "_upstreamResults": {"Up": upstream},
                    "api": {
                        "method": "POST",
                        "url": "https://custom.example.com/v1/chat/completions",
                        "json": {"messages": "{{prompt}}"},
                    },
                    "data": {
                        "type": "dataframe",
                        "columns": [
                            {
                                "label": "L",
                                "node": "Up",
                                "path": "items.rows.json.choices[0].message.content",
                            }
                        ],
                        "messages": [
                            {"role": "user", "content": "row {L}"},
                        ],
                    },
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _EchoTransport()
        result = _run(_executor(), task, transport, tmp_path)
        assert len(transport.requests) == 3
        assert len(result.items) == 2
        assert [len(item.rows) for item in result.items] == [2, 1]
        # Group 0 holds rows c0, c1; group 1 holds c2.
        group0 = {json.loads(r.prompt)[0]["content"] for r in result.items[0].rows}
        assert group0 == {"row c0", "row c1"}
        assert json.loads(result.items[1].rows[0].prompt)[0]["content"] == "row c2"
