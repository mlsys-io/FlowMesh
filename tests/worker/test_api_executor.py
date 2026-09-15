"""Tests for the endpoint-agnostic API executor auth and URL handling."""

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
                    "url": "https://api.example.com/v1/chat/completions",
                    "method": "POST",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
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


class TestAuth:
    def test_injects_bearer_from_credential_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_API_TOKEN", "secret-token")
        task = _task_message(auth={"credential_env": "LLM_API_TOKEN"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer secret-token"

    def test_custom_header_and_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "abc")
        task = _task_message(
            auth={"credential_env": "MY_KEY", "header": "X-API-Key", "scheme": "Token"}
        )
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["X-API-Key"] == "Token abc"

    def test_missing_credential_env_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_API_TOKEN", raising=False)
        task = _task_message(auth={"credential_env": "LLM_API_TOKEN"})
        with pytest.raises(ExecutionError, match="LLM_API_TOKEN"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_auth_not_mapping_rejected(self) -> None:
        task = _task_message(auth="Bearer x")
        with pytest.raises(ExecutionError, match="spec.api.auth must be a mapping"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_credential_env_must_be_nonempty_string(self) -> None:
        task = _task_message(auth={"credential_env": ""})
        with pytest.raises(ExecutionError, match="credential_env"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_explicit_authorization_header_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_API_TOKEN", "secret-token")
        task = _task_message(
            auth={"credential_env": "LLM_API_TOKEN"},
            headers={"Authorization": "Bearer explicit"},
        )
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer explicit"


class TestUrl:
    def test_url_required_when_no_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_BASE_URL", raising=False)
        task = _task_message(url=None)
        with pytest.raises(ExecutionError, match="spec.api.url or NEBULA_API_BASE_URL"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_url_falls_back_to_nebula_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(url=None)
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://nebula.example.com/v1/chat/completions"

    def test_explicit_authorization_header_allows_anonymous_call(self) -> None:
        task = _task_message(headers={"Authorization": "Bearer explicit"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer explicit"

    def test_auth_mode_none_allows_anonymous_call(self) -> None:
        task = _task_message(auth={"mode": "none"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert "Authorization" not in transport.request.headers


class TestLegacyNebulaPath:
    def test_no_auth_block_uses_nebula_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer nebula-token"

    def test_no_auth_block_and_no_token_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_TOKEN", raising=False)
        task = _task_message()
        with pytest.raises(ExecutionError, match="NEBULA_API_TOKEN"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())

    def test_credential_env_wins_over_nebula_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        monkeypatch.setenv("LLM_API_TOKEN", "general-token")
        task = _task_message(auth={"credential_env": "LLM_API_TOKEN"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer general-token"

    def test_spec_authorization_header_wins_over_nebula_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")
        task = _task_message(headers={"Authorization": "Bearer explicit"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer explicit"
