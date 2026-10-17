"""Workflow resource operations."""

import asyncio
import builtins
import json
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import yaml

from ..models.common import (
    TERMINAL_WORKFLOW_STATUSES,
    LogEntry,
    LogEvent,
    LogQueryResponse,
)
from ..models.workflows import (
    Workflow,
    WorkflowStatusView,
    WorkflowSubmitResponse,
    WorkflowValidateResponse,
)
from ..params import append_param, extend_params
from ._base import AsyncResource, SyncResource

WorkflowFormat = Literal["native", "n8n"]


def _serialize_workflow(
    workflow: str | dict[str, Any], workflow_format: WorkflowFormat
) -> str:
    if isinstance(workflow, str):
        return workflow
    if workflow_format == "n8n":
        return json.dumps(workflow)
    return yaml.safe_dump(workflow, sort_keys=False)


def _workflow_content_type(workflow_format: WorkflowFormat) -> str:
    return "application/json" if workflow_format == "n8n" else "text/plain"


def _list_params(
    workflow_id: str | None = None,
    status: str | list[str] | None = None,
    task_ids: str | list[str] | None = None,
    limit: int | None = None,
    query_params: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    append_param(params, "workflow_id", workflow_id)
    extend_params(params, "status", status)
    extend_params(params, "task_ids", task_ids)
    append_param(params, "limit", limit)
    if query_params:
        params.extend(query_params)
    return params


class Workflows(SyncResource):
    """Synchronous workflow operations."""

    def submit(
        self, workflow: str | dict[str, Any], workflow_format: WorkflowFormat = "native"
    ) -> WorkflowSubmitResponse:
        """Submit a workflow definition.

        Args:
            workflow: YAML/JSON workflow text or a structured workflow mapping.
            workflow_format: ``"native"`` (default) or ``"n8n"``.
        """
        workflow_content = _serialize_workflow(workflow, workflow_format)
        data = self._client._request(
            "POST",
            "/workflows",
            data=workflow_content,
            headers={
                "Content-Type": _workflow_content_type(workflow_format),
                "Workflow-Format": workflow_format,
            },
        )
        return WorkflowSubmitResponse.model_validate(data)

    def validate(
        self, workflow: str | dict[str, Any], workflow_format: WorkflowFormat = "native"
    ) -> WorkflowValidateResponse:
        """Validate a workflow definition without executing."""
        workflow_content = _serialize_workflow(workflow, workflow_format)
        data = self._client._request(
            "POST",
            "/workflows/validate",
            data=workflow_content,
            headers={
                "Content-Type": _workflow_content_type(workflow_format),
                "Workflow-Format": workflow_format,
            },
        )
        return WorkflowValidateResponse.model_validate(data)

    def retrieve(self, workflow_id: str) -> Workflow:
        """Retrieve workflow details by ID."""
        data = self._client._request("GET", f"/workflows/{workflow_id}")
        return Workflow.model_validate(data)

    def status(self, workflow_id: str) -> WorkflowStatusView:
        """Retrieve a workflow's status view, with task counts instead of task IDs.

        Prefer this over ``retrieve`` for polling."""
        data = self._client._request(
            "GET", f"/workflows/{workflow_id}", params=[("view", "status")]
        )
        return WorkflowStatusView.model_validate(data)

    def list(
        self,
        workflow_id: str | None = None,
        status: str | list[str] | None = None,
        task_ids: str | list[str] | None = None,
        limit: int | None = None,
        query_params: list[tuple[str, str]] | None = None,
    ) -> list[Workflow]:
        """List workflows with optional filters, oldest submission first.

        ``limit`` keeps only the newest matching workflows."""
        params = _list_params(
            workflow_id=workflow_id,
            status=status,
            task_ids=task_ids,
            limit=limit,
            query_params=query_params,
        )
        data = self._client._request("GET", "/workflows", params=params or None)
        return [Workflow.model_validate(w) for w in data]

    def list_statuses(
        self,
        workflow_id: str | None = None,
        status: str | builtins.list[str] | None = None,
        task_ids: str | builtins.list[str] | None = None,
        limit: int | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkflowStatusView]:
        """List the status views of workflows; takes the filters of ``list``."""
        params = _list_params(
            workflow_id=workflow_id,
            status=status,
            task_ids=task_ids,
            limit=limit,
            query_params=query_params,
        )
        params.append(("view", "status"))
        data = self._client._request("GET", "/workflows", params=params)
        return [WorkflowStatusView.model_validate(w) for w in data]

    def cancel(self, workflow_id: str) -> Workflow:
        """Cancel a running workflow."""
        data = self._client._request("POST", f"/workflows/{workflow_id}/cancel")
        return Workflow.model_validate(data)

    def get_logs(
        self,
        workflow_id: str,
        limit: int = 500,
        before: str | None = None,
        after: str | None = None,
    ) -> LogQueryResponse:
        """Query workflow logs with cursor-based pagination."""
        params: dict[str, str] = {"limit": str(limit)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        data = self._client._request(
            "GET", f"/workflows/{workflow_id}/logs", params=params
        )
        return LogQueryResponse.model_validate(data)

    def stream_logs(
        self, workflow_id: str, cursor: str | None = None
    ) -> Iterator[LogEntry]:
        """Stream workflow logs via SSE. Yields ``LogEntry`` objects."""
        for event_cursor, event_data in self._client._stream_sse(
            f"/workflows/{workflow_id}/logs/stream", cursor=cursor
        ):
            yield LogEntry(
                cursor=event_cursor or "",
                event=LogEvent.model_validate(event_data),
            )

    def download_logs(self, workflow_id: str, output_dir: Path) -> Iterator[Path]:
        """Download archived logs for all tasks in a workflow."""
        workflow = self.retrieve(workflow_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        for task_id in workflow.task_ids:
            out_path = output_dir / f"{task_id}-logs.jsonl"
            self._client._download(f"/results/{task_id}/logs", out_path)
            yield out_path

    def wait(
        self,
        workflow_id: str,
        interval: float = 2.0,
    ) -> Workflow:
        """Poll a workflow until it reaches a terminal state, then retrieve it."""
        self.wait_status(workflow_id, interval=interval)
        return self.retrieve(workflow_id)

    def wait_status(
        self,
        workflow_id: str,
        interval: float = 2.0,
    ) -> WorkflowStatusView:
        """Poll a workflow until it reaches a terminal state and return its status
        view."""
        while True:
            view = self.status(workflow_id)
            if view.status in TERMINAL_WORKFLOW_STATUSES:
                return view
            time.sleep(interval)


class AsyncWorkflows(AsyncResource):
    """Asynchronous workflow operations."""

    async def submit(
        self, workflow: str | dict[str, Any], workflow_format: WorkflowFormat = "native"
    ) -> WorkflowSubmitResponse:
        """Submit a workflow definition."""
        workflow_content = _serialize_workflow(workflow, workflow_format)
        data = await self._client._request(
            "POST",
            "/workflows",
            data=workflow_content,
            headers={
                "Content-Type": _workflow_content_type(workflow_format),
                "Workflow-Format": workflow_format,
            },
        )
        return WorkflowSubmitResponse.model_validate(data)

    async def validate(
        self, workflow: str | dict[str, Any], workflow_format: WorkflowFormat = "native"
    ) -> WorkflowValidateResponse:
        """Validate a workflow definition without executing."""
        workflow_content = _serialize_workflow(workflow, workflow_format)
        data = await self._client._request(
            "POST",
            "/workflows/validate",
            data=workflow_content,
            headers={
                "Content-Type": _workflow_content_type(workflow_format),
                "Workflow-Format": workflow_format,
            },
        )
        return WorkflowValidateResponse.model_validate(data)

    async def retrieve(self, workflow_id: str) -> Workflow:
        """Retrieve workflow details by ID."""
        data = await self._client._request("GET", f"/workflows/{workflow_id}")
        return Workflow.model_validate(data)

    async def status(self, workflow_id: str) -> WorkflowStatusView:
        """Retrieve a workflow's status view, with task counts instead of task IDs.

        Prefer this over ``retrieve`` for polling."""
        data = await self._client._request(
            "GET", f"/workflows/{workflow_id}", params=[("view", "status")]
        )
        return WorkflowStatusView.model_validate(data)

    async def list(
        self,
        workflow_id: str | None = None,
        status: str | list[str] | None = None,
        task_ids: str | list[str] | None = None,
        limit: int | None = None,
        query_params: list[tuple[str, str]] | None = None,
    ) -> list[Workflow]:
        """List workflows with optional filters, oldest submission first.

        ``limit`` keeps only the newest matching workflows."""
        params = _list_params(
            workflow_id=workflow_id,
            status=status,
            task_ids=task_ids,
            limit=limit,
            query_params=query_params,
        )
        data = await self._client._request("GET", "/workflows", params=params or None)
        return [Workflow.model_validate(w) for w in data]

    async def list_statuses(
        self,
        workflow_id: str | None = None,
        status: str | builtins.list[str] | None = None,
        task_ids: str | builtins.list[str] | None = None,
        limit: int | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkflowStatusView]:
        """List the status views of workflows; takes the filters of ``list``."""
        params = _list_params(
            workflow_id=workflow_id,
            status=status,
            task_ids=task_ids,
            limit=limit,
            query_params=query_params,
        )
        params.append(("view", "status"))
        data = await self._client._request("GET", "/workflows", params=params)
        return [WorkflowStatusView.model_validate(w) for w in data]

    async def cancel(self, workflow_id: str) -> Workflow:
        """Cancel a running workflow."""
        data = await self._client._request("POST", f"/workflows/{workflow_id}/cancel")
        return Workflow.model_validate(data)

    async def get_logs(
        self,
        workflow_id: str,
        limit: int = 500,
        before: str | None = None,
        after: str | None = None,
    ) -> LogQueryResponse:
        """Query workflow logs with cursor-based pagination."""
        params: dict[str, str] = {"limit": str(limit)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        data = await self._client._request(
            "GET", f"/workflows/{workflow_id}/logs", params=params
        )
        return LogQueryResponse.model_validate(data)

    async def stream_logs(
        self, workflow_id: str, cursor: str | None = None
    ) -> AsyncIterator[LogEntry]:
        """Stream workflow logs via SSE. Yields ``LogEntry`` objects."""
        async for event_cursor, event_data in self._client._stream_sse(
            f"/workflows/{workflow_id}/logs/stream", cursor=cursor
        ):
            yield LogEntry(
                cursor=event_cursor or "",
                event=LogEvent.model_validate(event_data),
            )

    async def download_logs(
        self, workflow_id: str, output_dir: Path
    ) -> AsyncIterator[Path]:
        """Download archived logs for all tasks in a workflow."""
        workflow = await self.retrieve(workflow_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        for task_id in workflow.task_ids:
            out_path = output_dir / f"{task_id}-logs.jsonl"
            await self._client._download(f"/results/{task_id}/logs", out_path)
            yield out_path

    async def wait(self, workflow_id: str, interval: float = 2.0) -> Workflow:
        """Poll a workflow until it reaches a terminal state, then retrieve it."""
        await self.wait_status(workflow_id, interval=interval)
        return await self.retrieve(workflow_id)

    async def wait_status(
        self, workflow_id: str, interval: float = 2.0
    ) -> WorkflowStatusView:
        """Poll a workflow until it reaches a terminal state and return its status
        view."""
        while True:
            view = await self.status(workflow_id)
            if view.status in TERMINAL_WORKFLOW_STATUSES:
                return view
            await asyncio.sleep(interval)
