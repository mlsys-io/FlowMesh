import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.app_state import get_worker_registry
from server.auth.security import PrincipalContext, authenticate_connection
from server.registries.worker import Worker, cordon_member
from server.routers.v1 import workers as workers_router
from shared.schemas.worker import WorkerStatus

PREFIX = "/api/v1"


def _worker(worker_id: str, alias: str | None = "alpha") -> Worker:
    return Worker(
        id=worker_id,
        alias=alias,
        namespace="default",
        cluster="default",
        node_id="nde-1",
        node_alias="node",
        status=WorkerStatus.IDLE,
    )


def _registry(workers: list[Worker]) -> MagicMock:
    by_id = {w.id: w for w in workers}
    registry = MagicMock()
    registry.get_worker_async = AsyncMock(side_effect=by_id.get)
    registry.is_worker_stale_async = AsyncMock(return_value=False)
    registry.cordoned_members_async = AsyncMock(
        return_value={cordon_member("node", "alpha")}
    )
    registry.set_cordon_async = AsyncMock(return_value=True)
    return registry


def _client(registry: MagicMock) -> AsyncClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.workers_router")
    app.include_router(workers_router.router, prefix=PREFIX)
    app.dependency_overrides[get_worker_registry] = lambda: registry
    app.dependency_overrides[authenticate_connection] = lambda: MagicMock(
        spec=PrincipalContext
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.anyio
async def test_cordon_keys_on_node_alias_and_worker_alias() -> None:
    registry = _registry([_worker("wkr-1")])
    async with _client(registry) as ac:
        resp = await ac.post(f"{PREFIX}/workers/wkr-1/cordon")
    assert resp.status_code == 200
    assert resp.json() == {
        "node_alias": "node",
        "alias": "alpha",
        "cordoned": True,
        "changed": True,
    }


@pytest.mark.anyio
async def test_cordon_refuses_a_worker_without_an_alias() -> None:
    registry = _registry([_worker("wkr-1", alias=None)])
    async with _client(registry) as ac:
        resp = await ac.post(f"{PREFIX}/workers/wkr-1/cordon")
    assert resp.status_code == 409
    registry.set_cordon_async.assert_not_called()


@pytest.mark.anyio
async def test_cordon_unknown_worker_is_404() -> None:
    async with _client(_registry([])) as ac:
        resp = await ac.post(f"{PREFIX}/workers/wkr-9/cordon")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_worker_reports_cordon_state() -> None:
    registry = _registry([_worker("wkr-1")])
    async with _client(registry) as ac:
        resp = await ac.get(f"{PREFIX}/workers/wkr-1")
    assert resp.status_code == 200
    assert resp.json()["cordoned"] is True


@pytest.mark.anyio
async def test_remove_cordon_does_not_need_a_registered_worker() -> None:
    registry = _registry([])
    async with _client(registry) as ac:
        resp = await ac.delete(f"{PREFIX}/workers/cordoned/node/alpha")
    assert resp.status_code == 200
    assert resp.json() == {
        "node_alias": "node",
        "alias": "alpha",
        "cordoned": False,
        "changed": True,
    }
