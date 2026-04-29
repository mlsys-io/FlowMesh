"""Workflow profile (trace analysis) resource."""

from typing import Any

from ._base import AsyncResource, SyncResource


class Profile(SyncResource):
    """Synchronous trace analyzer access."""

    def fetch(self, workflow_id: str) -> dict[str, Any]:
        """Return the analyzer's structured `ProfileSummary` for a workflow."""
        return self._client._request("GET", f"/workflows/{workflow_id}/profile")


class AsyncProfile(AsyncResource):
    """Asynchronous trace analyzer access."""

    async def fetch(self, workflow_id: str) -> dict[str, Any]:
        return await self._client._request(
            "GET", f"/workflows/{workflow_id}/profile"
        )
