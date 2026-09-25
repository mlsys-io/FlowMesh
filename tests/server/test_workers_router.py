import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.app_state import get_worker_registry
from server.auth.security import PrincipalContext, authenticate_connection
from server.registries.worker import Worker, cordon_member
from server.routers.v1 import workers as workers_router
from server.schemas.worker import WorkerCordon
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
    registry.live_worker_ids_async = AsyncMock(
        side_effect=lambda cordon: [
            w.id
            for w in workers
            if (w.node_alias, w.alias) == (cordon.node_alias, cordon.alias)
        ]
    )
    registry.list_cordons_async = AsyncMock(
        return_value=[WorkerCordon(node_alias="node", alias="alpha")]
    )
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


async def _post(registry: MagicMock, route: str, body: dict[str, Any]) -> Any:
    async with _client(registry) as ac:
        return await ac.post(f"{PREFIX}/workers/{route}", json=body)


@pytest.mark.anyio
async def test_cordon_by_worker_id_keys_on_its_aliases() -> None:
    registry = _registry([_worker("wkr-1")])
    resp = await _post(registry, "cordon", {"worker_id": "wkr-1"})
    assert resp.status_code == 200
    assert resp.json() == {
        "node_alias": "node",
        "alias": "alpha",
        "cordoned": True,
        "changed": True,
        "worker_ids": ["wkr-1"],
    }
    registry.set_cordon_async.assert_awaited_once_with(
        WorkerCordon(node_alias="node", alias="alpha"), cordoned=True
    )


@pytest.mark.anyio
async def test_cordon_by_alias_needs_no_registered_worker() -> None:
    registry = _registry([])
    resp = await _post(registry, "cordon", {"node_alias": "node", "alias": "beta"})
    assert resp.status_code == 200
    assert resp.json() == {
        "node_alias": "node",
        "alias": "beta",
        "cordoned": True,
        "changed": True,
        "worker_ids": [],
    }


@pytest.mark.anyio
async def test_uncordon_by_alias_needs_no_registered_worker() -> None:
    registry = _registry([])
    resp = await _post(registry, "uncordon", {"node_alias": "node", "alias": "alpha"})
    assert resp.status_code == 200
    assert resp.json()["cordoned"] is False
    registry.set_cordon_async.assert_awaited_once_with(
        WorkerCordon(node_alias="node", alias="alpha"), cordoned=False
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"node_alias": "node"},
        {"alias": "alpha"},
        {"worker_id": "wkr-1", "node_alias": "node", "alias": "alpha"},
        {"worker_id": "wkr-1", "alias": "alpha"},
    ],
)
async def test_cordon_requires_exactly_one_selector(body: dict[str, Any]) -> None:
    registry = _registry([_worker("wkr-1")])
    resp = await _post(registry, "cordon", body)
    assert resp.status_code == 422
    registry.set_cordon_async.assert_not_called()


@pytest.mark.anyio
async def test_cordon_unknown_worker_id_is_404() -> None:
    registry = _registry([])
    resp = await _post(registry, "cordon", {"worker_id": "wkr-9"})
    assert resp.status_code == 404
    registry.set_cordon_async.assert_not_called()


@pytest.mark.anyio
async def test_cordon_worker_without_an_alias_is_409() -> None:
    registry = _registry([_worker("wkr-1", alias=None)])
    resp = await _post(registry, "cordon", {"worker_id": "wkr-1"})
    assert resp.status_code == 409
    registry.set_cordon_async.assert_not_called()


@pytest.mark.anyio
async def test_list_cordons() -> None:
    async with _client(_registry([])) as ac:
        resp = await ac.get(f"{PREFIX}/workers/cordons")
    assert resp.status_code == 200
    assert resp.json() == [{"node_alias": "node", "alias": "alpha"}]


@pytest.mark.anyio
async def test_get_worker_reports_cordon_state() -> None:
    async with _client(_registry([_worker("wkr-1")])) as ac:
        resp = await ac.get(f"{PREFIX}/workers/wkr-1")
    assert resp.status_code == 200
    assert resp.json()["cordoned"] is True
