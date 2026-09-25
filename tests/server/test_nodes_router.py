"""Tests for the /api/v1/nodes router."""

import logging
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.app_state import get_logger, get_node_registry
from server.auth.security import authenticate_connection
from server.registries import NodeAliasInUseError
from server.routers.v1 import nodes as nodes_router

PREFIX = "/api/v1"
NODE_INFO = {
    "namespace": "ns",
    "cluster": "cl",
    "alias": "gpu-a",
    "version": "0.1.0",
    "started_at": "2026-09-25T00:00:00Z",
    "tags": [],
    "last_seen": "2026-09-25T00:00:00Z",
    "max_gpu_count": 0,
}


class _Registry:
    def __init__(self, holder: str | None) -> None:
        self.holder = holder

    async def register_node_async(self, node_info: Any) -> str:
        if self.holder is not None:
            raise NodeAliasInUseError(node_info.alias, self.holder)
        return "nde-1"


async def _allow(*args: Any, **kwargs: Any) -> None:
    return None


def _make_app(registry: _Registry) -> FastAPI:
    app = FastAPI()
    app.include_router(nodes_router.router, prefix=PREFIX)
    app.dependency_overrides[authenticate_connection] = lambda: None
    app.dependency_overrides[get_node_registry] = lambda: registry
    app.dependency_overrides[get_logger] = lambda: logging.getLogger("test.nodes")
    return app


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("holder", "status_code"), [(None, 201), ("nde-7", 409)], ids=["free", "held"]
)
async def test_register_node(
    monkeypatch: pytest.MonkeyPatch, holder: str | None, status_code: int
) -> None:
    monkeypatch.setattr(nodes_router, "require_permission", _allow)
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(_Registry(holder))), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/nodes/register", json=NODE_INFO)

    assert resp.status_code == status_code
    if holder is not None:
        assert resp.json()["detail"] == (
            "node alias 'gpu-a' is held by live node nde-7; set a distinct NODE_ALIAS"
        )
