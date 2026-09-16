"""Tests for the API executor defaults, credential handling, and model injection."""

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


class TestDefaults:
    def test_default_url_is_lum_id_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "token")
        task = _task_message(url=None)
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.url == "https://lum.id/llm/v1/chat/completions"

    def test_default_model_injected_when_body_has_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        sent = _sent_json(transport.request)
        assert sent["model"] == "deepseek-v4-flash"

    def test_explicit_model_left_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "token")
        task = _task_message(body={"model": "gpt-4o", "messages": []})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        sent = _sent_json(transport.request)
        assert sent["model"] == "gpt-4o"


class TestCredential:
    def test_explicit_authorization_header_left_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "worker-token")
        task = _task_message(headers={"Authorization": "Bearer explicit"})
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer explicit"

    def test_nebula_token_used_when_no_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEBULA_API_TOKEN", "worker-token")
        task = _task_message()
        transport = _RecordingTransport()
        _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert transport.request is not None
        assert transport.request.headers["Authorization"] == "Bearer worker-token"

    def test_missing_credential_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBULA_API_TOKEN", raising=False)
        task = _task_message()
        with pytest.raises(ExecutionError, match="NEBULA_API_TOKEN"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())


def _sent_json(request: httpx.Request) -> dict:
    import json

    return json.loads(request.content.decode("utf-8"))
