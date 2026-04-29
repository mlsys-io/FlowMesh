"""Workflow lineage log resources (events, assets, lineage)."""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import httpx

from .._base_client import _make_url, _raise_for_stream_status
from ..exceptions import FlowMeshConnectionError
from ._base import AsyncResource, SyncResource

type LineageKind = Literal["events", "assets", "lineage"]


def _iter_jsonl_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


async def _aiter_jsonl_lines(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    async for line in lines:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


class Logs(SyncResource):
    """Synchronous workflow-scoped lineage operations."""

    def fetch(self, workflow_id: str, kind: LineageKind) -> Iterator[dict[str, Any]]:
        """Yield JSONL rows for `events`, `assets`, or `lineage`."""
        url = _make_url(self._client.base_url, f"/workflows/{workflow_id}/logs/{kind}")
        try:
            with self._client._http.stream("GET", url) as response:
                _raise_for_stream_status(response, "GET")
                yield from _iter_jsonl_lines(response.iter_lines())
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def fetch_events(self, workflow_id: str) -> Iterator[dict[str, Any]]:
        return self.fetch(workflow_id, "events")

    def fetch_assets(self, workflow_id: str) -> Iterator[dict[str, Any]]:
        return self.fetch(workflow_id, "assets")

    def fetch_lineage(self, workflow_id: str) -> Iterator[dict[str, Any]]:
        return self.fetch(workflow_id, "lineage")


class AsyncLogs(AsyncResource):
    """Asynchronous workflow-scoped lineage operations."""

    async def fetch(
        self, workflow_id: str, kind: LineageKind
    ) -> AsyncIterator[dict[str, Any]]:
        url = _make_url(self._client.base_url, f"/workflows/{workflow_id}/logs/{kind}")
        try:
            async with self._client._http.stream("GET", url) as response:
                from .._base_client import _raise_for_stream_status_async

                await _raise_for_stream_status_async(response, "GET")
                async for row in _aiter_jsonl_lines(response.aiter_lines()):
                    yield row
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def fetch_events(self, workflow_id: str) -> AsyncIterator[dict[str, Any]]:
        async for row in self.fetch(workflow_id, "events"):
            yield row

    async def fetch_assets(self, workflow_id: str) -> AsyncIterator[dict[str, Any]]:
        async for row in self.fetch(workflow_id, "assets"):
            yield row

    async def fetch_lineage(self, workflow_id: str) -> AsyncIterator[dict[str, Any]]:
        async for row in self.fetch(workflow_id, "lineage"):
            yield row
