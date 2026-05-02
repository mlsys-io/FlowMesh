"""Workflow trace resource — fetch raw rows or run the analyzer."""

from collections.abc import AsyncIterator, Iterator
from enum import StrEnum

import httpx

from .._base_client import (
    _make_url,
    _raise_for_stream_status,
    _raise_for_stream_status_async,
)
from .._jsonl import aparse_jsonl_lines, parse_jsonl_lines
from ..exceptions import FlowMeshConnectionError
from ..models.trace import ProfileSummary
from ._base import AsyncResource, SyncResource


class TraceKind(StrEnum):
    """Trace row kind. Members serialize as their values."""

    SPANS = "spans"
    ASSETS = "assets"
    LINEAGE = "lineage"


class Trace(SyncResource):
    """Synchronous workflow trace operations."""

    def fetch(self, workflow_id: str, kind: TraceKind) -> Iterator[dict]:
        """Yield JSONL rows for `spans`, `assets`, or `lineage`."""
        url = _make_url(self._client.base_url, f"/trace/{workflow_id}/{kind}")
        try:
            with self._client._http.stream("GET", url) as response:
                _raise_for_stream_status(response, "GET")
                yield from parse_jsonl_lines(response.iter_lines())
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def analyze(self, workflow_id: str) -> ProfileSummary:
        """Run the trace analyzer and return a parsed `ProfileSummary`."""
        return ProfileSummary.model_validate(
            self._client._request("GET", f"/trace/{workflow_id}/analyze")
        )


class AsyncTrace(AsyncResource):
    """Asynchronous workflow trace operations."""

    async def fetch(self, workflow_id: str, kind: TraceKind) -> AsyncIterator[dict]:
        url = _make_url(self._client.base_url, f"/trace/{workflow_id}/{kind}")
        try:
            async with self._client._http.stream("GET", url) as response:
                await _raise_for_stream_status_async(response, "GET")
                async for row in aparse_jsonl_lines(response.aiter_lines()):
                    yield row
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def analyze(self, workflow_id: str) -> ProfileSummary:
        return ProfileSummary.model_validate(
            await self._client._request("GET", f"/trace/{workflow_id}/analyze")
        )
