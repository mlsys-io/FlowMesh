import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, status
from httpx import ASGITransport, AsyncClient
from lumid_hooks import ResourceRef

from server.app_state import get_worker_registry
from server.auth.security import PrincipalContext, authenticate_connection
from server.hooks import PERMISSION_CHECKERS
from server.registries.worker import Worker
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


class _NonAdminChecker:
    """Denies admin and scopes worker access to `wkr-1`."""

    name = "non-admin"

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        if action == "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return frozenset({"wkr-1"})


@pytest.fixture
def non_admin() -> Iterator[None]:
    PERMISSION_CHECKERS.append(_NonAdminChecker())
    try:
        yield
    finally:
        PERMISSION_CHECKERS.clear()


def _registry(workers: list[Worker]) -> MagicMock:
    by_id = {w.id: w for w in workers}
    registry = MagicMock()
    registry.get_worker_async = AsyncMock(side_effect=by_id.get)
    registry.get_workers_async = AsyncMock(
        side_effect=lambda ids: [by_id.get(i) for i in ids]
    )
    registry.is_worker_stale_async = AsyncMock(return_value=False)
    registry.is_cordoned_async = AsyncMock(return_value=True)
    registry.set_cordon_async = AsyncMock(return_value=True)
    registry.live_worker_ids_for_cordon_async = AsyncMock(
        side_effect=lambda cordon: [
            w.id
            for w in workers
            if (w.node_alias, w.alias) == (cordon.node_alias, cordon.alias)
        ]
    )
    registry.list_cordons_async = AsyncMock(
        return_value=[
            WorkerCordon(node_alias="node", alias="alpha"),
            WorkerCordon(node_alias="node", alias="beta"),
        ]
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
        {"node_alias": "node", "alias": ""},
        {"node_alias": "", "alias": "alpha"},
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
@pytest.mark.parametrize("route", ["cordon", "uncordon"])
@pytest.mark.usefixtures("non_admin")
async def test_selecting_by_alias_requires_admin(route: str) -> None:
    registry = _registry([_worker("wkr-1")])
    resp = await _post(registry, route, {"node_alias": "node", "alias": "alpha"})
    assert resp.status_code == 403
    registry.set_cordon_async.assert_not_called()


@pytest.mark.anyio
@pytest.mark.usefixtures("non_admin")
async def test_non_admin_cordons_an_accessible_worker_by_id() -> None:
    registry = _registry([_worker("wkr-1")])
    resp = await _post(registry, "cordon", {"worker_id": "wkr-1"})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_list_cordons_unfiltered_includes_keys_with_no_worker() -> None:
    async with _client(_registry([])) as ac:
        resp = await ac.get(f"{PREFIX}/workers/cordons")
    assert resp.status_code == 200
    assert resp.json() == [
        {"node_alias": "node", "alias": "alpha"},
        {"node_alias": "node", "alias": "beta"},
    ]


@pytest.mark.anyio
@pytest.mark.usefixtures("non_admin")
async def test_list_cordons_is_limited_to_accessible_workers() -> None:
    registry = _registry([_worker("wkr-1", "alpha"), _worker("wkr-2", "beta")])
    async with _client(registry) as ac:
        resp = await ac.get(f"{PREFIX}/workers/cordons")
    assert resp.status_code == 200
    assert resp.json() == [{"node_alias": "node", "alias": "alpha"}]


@pytest.mark.anyio
async def test_get_worker_reports_cordon_state() -> None:
    async with _client(_registry([_worker("wkr-1")])) as ac:
        resp = await ac.get(f"{PREFIX}/workers/wkr-1")
    assert resp.status_code == 200
    assert resp.json()["cordoned"] is True
