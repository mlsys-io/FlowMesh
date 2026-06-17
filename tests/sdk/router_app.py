"""Lightweight FastAPI app for deriving server route paths in SDK tests."""

from functools import lru_cache

from fastapi import FastAPI

from server.routers.health import router as health_router
from server.routers.v1.nodes import router as nodes_router
from server.routers.v1.system import router as system_router
from server.routers.v1.tasks import router as tasks_router
from server.routers.v1.workers import router as workers_router
from server.routers.v1.workflows import router as workflows_router

TEST_BASE_URL = "http://test-server:8000"


@lru_cache(maxsize=1)
def build_router_app() -> FastAPI:
    """Build a minimal app containing only the routers needed by SDK tests."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(system_router, prefix="/api/v1")
    app.include_router(nodes_router, prefix="/api/v1")
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(workers_router, prefix="/api/v1")
    app.include_router(workflows_router, prefix="/api/v1")
    return app


def route_url(name: str, **path_params: str) -> str:
    """Return the full test URL for a named server route."""
    path = build_router_app().url_path_for(name, **path_params)
    return f"{TEST_BASE_URL}{path}"
