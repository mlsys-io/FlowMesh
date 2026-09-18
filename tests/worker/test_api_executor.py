"""Tests for the API executor url override and Nebula credential handling."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from worker.executors.api_executor import APIExecutor
from worker.executors.base_executor import ExecutionError


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api",
        "workflow_id": "wf-1",
        "owner_id": "owner",
        "assigned_worker": "worker-1",
        "dispatched_at": "2026-03-22T00:00:00Z",
        "task": {
            "apiVersion": "mloc/v1",
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


def _run(
    executor: APIExecutor, task: WorkerTaskMessage, transport: _RecordingTransport
) -> None:
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
