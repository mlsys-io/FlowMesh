"""Tests for the API executor's batch mode (one task, N row-aligned requests)."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from worker.executors.api_executor import APIExecutor
from worker.executors.base_executor import ExecutionError


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api-batch",
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
                    "body": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
                    **spec_updates,
                },
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


class _RecordingTransport(httpx.MockTransport):
    """MockTransport that records every request it served, in order."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"total_tokens": 3},
            },
        )


def _run(
    executor: APIExecutor, task: WorkerTaskMessage, transport: httpx.MockTransport
):
    with patch.object(
        APIExecutor, "_get_client", return_value=httpx.Client(transport=transport)
    ):
        return executor.run(task, Path("/tmp/out"))


def _batch_task(items: list[object]) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-api-batch",
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
                    "body": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
                },
                "data": {"type": "list", "items": items},
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


class TestBatch:
    @pytest.fixture(autouse=True)
    def _nebula_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEBULA_API_BASE_URL", "https://nebula.example.com")
        monkeypatch.setenv("NEBULA_API_TOKEN", "nebula-token")

    def test_issues_one_request_per_row_in_order(self) -> None:
        task = _batch_task(["first", "second", "third"])
        transport = _RecordingTransport()
        result = _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert len(transport.requests) == 3
        # Row order is preserved: request i carries row i's prompt.
        for idx, prompt in enumerate(["first", "second", "third"]):
            body = transport.requests[idx].read()
            assert prompt.encode() in body
        assert [item.index for item in result.items] == [0, 1, 2]
        assert [item.text for item in result.items] == ["hello"] * 3

    def test_single_row_batches_to_one_item(self) -> None:
        task = _batch_task(["only"])
        transport = _RecordingTransport()
        result = _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert len(transport.requests) == 1
        assert len(result.items) == 1
        assert result.items[0].index == 0
        assert result.items[0].prompt == "only"

    def test_row_failure_fails_whole_task_without_shifting(self) -> None:
        class _FailSecond(httpx.MockTransport):
            def __init__(self) -> None:
                self.requests: list[httpx.Request] = []
                super().__init__(self._handler)

            def _handler(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                if len(self.requests) == 2:
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

        task = _batch_task(["a", "b", "c"])
        transport = _FailSecond()
        with pytest.raises(ExecutionError, match="row 1"):
            _run(APIExecutor.__new__(APIExecutor), task, transport)
        # The failing row aborts the task; no partial result is returned.
        assert len(transport.requests) == 2

    def test_placeholder_not_required_for_scalar_body(self) -> None:
        """A batch task whose body has no placeholder still issues N requests."""
        payload = {
            "task_id": "task-api-batch",
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
                        "body": {"messages": [{"role": "user", "content": "static"}]},
                    },
                    "data": {"type": "list", "items": ["a", "b"]},
                },
            },
        }
        task = WorkerTaskMessage.model_validate(payload)
        transport = _RecordingTransport()
        result = _run(APIExecutor.__new__(APIExecutor), task, transport)
        assert len(transport.requests) == 2
        assert len(result.items) == 2

    def test_no_rows_raises(self) -> None:
        task = _batch_task([])
        with pytest.raises(ExecutionError, match="no rows"):
            _run(APIExecutor.__new__(APIExecutor), task, _RecordingTransport())
