"""Worker resource operations."""

import builtins
from urllib.parse import quote

from ..models.workers import WorkerCordon, WorkerCordonResult, WorkerInfo
from ..params import append_param, extend_params
from ._base import AsyncResource, SyncResource


def _cordon_path(node_alias: str, alias: str) -> str:
    return f"/workers/cordoned/{quote(node_alias, safe='')}/{quote(alias, safe='')}"


class Workers(SyncResource):
    """Synchronous worker operations."""

    def retrieve(self, worker_id: str) -> WorkerInfo:
        """Retrieve worker details by ID."""
        data = self._client._request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate(data)

    def list(
        self,
        worker_id: str | None = None,
        alias: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        status: str | builtins.list[str] | None = None,
        tags: str | builtins.list[str] | None = None,
        stale: bool | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkerInfo]:
        """List workers with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", worker_id)
        append_param(params, "alias", alias)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        extend_params(params, "status", status)
        extend_params(params, "tags", tags)
        append_param(params, "stale", stale)
        if query_params:
            params.extend(query_params)
        data = self._client._request("GET", "/workers", params=params or None)
        return [WorkerInfo.model_validate(w) for w in data]

    def cordon(self, worker_id: str) -> WorkerCordonResult:
        """Stop offering new tasks to a worker without stopping it."""
        data = self._client._request("POST", f"/workers/{worker_id}/cordon")
        return WorkerCordonResult.model_validate(data)

    def uncordon(self, worker_id: str) -> WorkerCordonResult:
        """Allow a cordoned worker to receive tasks again."""
        data = self._client._request("POST", f"/workers/{worker_id}/uncordon")
        return WorkerCordonResult.model_validate(data)

    def list_cordons(self) -> builtins.list[WorkerCordon]:
        """List active cordons."""
        data = self._client._request("GET", "/workers/cordoned")
        return [WorkerCordon.model_validate(c) for c in data]

    def remove_cordon(self, node_alias: str, alias: str) -> WorkerCordonResult:
        """Remove a cordon by node alias and worker alias."""
        data = self._client._request("DELETE", _cordon_path(node_alias, alias))
        return WorkerCordonResult.model_validate(data)


class AsyncWorkers(AsyncResource):
    """Asynchronous worker operations."""

    async def retrieve(self, worker_id: str) -> WorkerInfo:
        """Retrieve worker details by ID."""
        data = await self._client._request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate(data)

    async def list(
        self,
        worker_id: str | None = None,
        alias: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        status: str | builtins.list[str] | None = None,
        tags: str | builtins.list[str] | None = None,
        stale: bool | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkerInfo]:
        """List workers with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", worker_id)
        append_param(params, "alias", alias)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        extend_params(params, "status", status)
        extend_params(params, "tags", tags)
        append_param(params, "stale", stale)
        if query_params:
            params.extend(query_params)
        data = await self._client._request("GET", "/workers", params=params or None)
        return [WorkerInfo.model_validate(w) for w in data]

    async def cordon(self, worker_id: str) -> WorkerCordonResult:
        """Stop offering new tasks to a worker without stopping it."""
        data = await self._client._request("POST", f"/workers/{worker_id}/cordon")
        return WorkerCordonResult.model_validate(data)

    async def uncordon(self, worker_id: str) -> WorkerCordonResult:
        """Allow a cordoned worker to receive tasks again."""
        data = await self._client._request("POST", f"/workers/{worker_id}/uncordon")
        return WorkerCordonResult.model_validate(data)

    async def list_cordons(self) -> builtins.list[WorkerCordon]:
        """List active cordons."""
        data = await self._client._request("GET", "/workers/cordoned")
        return [WorkerCordon.model_validate(c) for c in data]

    async def remove_cordon(self, node_alias: str, alias: str) -> WorkerCordonResult:
        """Remove a cordon by node alias and worker alias."""
        data = await self._client._request("DELETE", _cordon_path(node_alias, alias))
        return WorkerCordonResult.model_validate(data)
