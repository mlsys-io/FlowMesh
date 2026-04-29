"""Workflow profile (trace analysis) resource."""

from typing import Any

from shared.profile import ProfileSummary

from ._base import AsyncResource, SyncResource


class Profile(SyncResource):
    """Synchronous trace analyzer access."""

    def fetch(self, workflow_id: str) -> dict[str, Any]:
        """Raw `ProfileSummary` payload (dict).

        Use `fetch_summary` for a parsed Pydantic model, or the helpers in
        `flowmesh.profile_views` for dataframe / mermaid views.
        """
        return self._client._request("GET", f"/workflows/{workflow_id}/profile")

    def fetch_summary(self, workflow_id: str) -> ProfileSummary:
        """Return a parsed `ProfileSummary` Pydantic model."""
        return ProfileSummary.model_validate(self.fetch(workflow_id))


class AsyncProfile(AsyncResource):
    """Asynchronous trace analyzer access."""

    async def fetch(self, workflow_id: str) -> dict[str, Any]:
        return await self._client._request("GET", f"/workflows/{workflow_id}/profile")

    async def fetch_summary(self, workflow_id: str) -> ProfileSummary:
        return ProfileSummary.model_validate(await self.fetch(workflow_id))
