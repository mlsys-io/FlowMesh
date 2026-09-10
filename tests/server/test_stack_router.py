"""Tests for the /api/v1/stack/workers router (IPC command channel)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.routers.v1 import stack as stack_router
from server.routers.v1._command import _STATUS_BY_ERROR_CODE
from shared.schemas.command import CommandErrorCode, CommandResponse

PREFIX = "/api/v1"


def _ok(data: dict | None = None) -> CommandResponse:
    return CommandResponse(command_id="test", success=True, data=data)


def _error(
    message: str = "fail", error_code: CommandErrorCode | None = None
) -> CommandResponse:
    return CommandResponse(
        command_id="test", success=False, message=message, error_code=error_code
    )


def _make_app(supervisor: MagicMock) -> FastAPI:
    app = FastAPI()
    app.state.supervisor = supervisor
    app.state.node_id = "nod-test"
    app.state.logger = logging.getLogger("test.stack_router")
    app.include_router(stack_router.router, prefix=PREFIX)
    return app


def _mock_supervisor(response: CommandResponse | None = None) -> MagicMock:
    sv = MagicMock()
    sv.exec_cmd = AsyncMock(return_value=response or _ok(data={}))
    return sv


_WORKERS = [
    {"id": "wkr-1", "name": "w1", "provider": "docker", "status": "RUNNING"},
    {"id": "wkr-2", "name": "w2", "provider": "vastai", "status": "STOPPED"},
]


# ------------------------------------------------------------------ #
# list / get workers
# ------------------------------------------------------------------ #


@pytest.mark.anyio
async def test_list_workers() -> None:
    sv = _mock_supervisor(_ok(data={"workers": _WORKERS}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.anyio
async def test_list_workers_empty() -> None:
    sv = _mock_supervisor(_ok(data={"workers": []}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
async def test_get_worker_found() -> None:
    sv = _mock_supervisor(_ok(data={"workers": _WORKERS}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers/w1")
    assert resp.status_code == 200
    assert resp.json()["name"] == "w1"


@pytest.mark.anyio
async def test_get_worker_not_found() -> None:
    sv = _mock_supervisor(_ok(data={"workers": []}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers/nope")
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# create worker
# ------------------------------------------------------------------ #


@pytest.mark.anyio
async def test_create_worker() -> None:
    sv = _mock_supervisor(
        _ok(
            data={
                "id": "wkr-3",
                "name": "new",
                "provider": "docker",
                "status": "RUNNING",
            }
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers", json={"provider": "docker"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new"


@pytest.mark.anyio
async def test_create_worker_fails() -> None:
    sv = _mock_supervisor(_error("bad config"))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers", json={"provider": "docker"})
    assert resp.status_code == 500
    assert "bad config" in resp.json()["detail"]


@pytest.mark.anyio
async def test_create_worker_provider_unavailable() -> None:
    sv = _mock_supervisor(
        _error(
            "Worker provider 'docker' is not available on this node; "
            "available providers: external",
            error_code=CommandErrorCode.PROVIDER_UNAVAILABLE,
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers", json={"provider": "docker"})
    assert resp.status_code == 409
    assert "docker" in resp.json()["detail"]
    assert "external" in resp.json()["detail"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        (CommandErrorCode.INVALID_PAYLOAD, 400),
        (CommandErrorCode.PROVIDER_UNAVAILABLE, 409),
        (CommandErrorCode.NOT_READY, 503),
        (CommandErrorCode.CANCELLED, 503),
        (CommandErrorCode.UNKNOWN_COMMAND, 501),
        (CommandErrorCode.INTERNAL, 500),
        (None, 500),
    ],
)
async def test_error_code_maps_to_http_status(
    error_code: CommandErrorCode | None, expected_status: int
) -> None:
    sv = _mock_supervisor(_error("boom", error_code=error_code))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers", json={"provider": "docker"})
    assert resp.status_code == expected_status
    assert resp.json()["detail"] == "boom"


def test_every_error_code_has_a_status() -> None:
    assert set(_STATUS_BY_ERROR_CODE) == set(CommandErrorCode)


@pytest.mark.anyio
async def test_get_providers() -> None:
    sv = _mock_supervisor(_ok(data={"providers": ["docker", "external"]}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers/providers")
    assert resp.status_code == 200
    assert resp.json() == {"providers": ["docker", "external"]}
    cmd = sv.exec_cmd.call_args[0][0]
    assert cmd.command.value == "GET_PROVIDERS"


# ------------------------------------------------------------------ #
# start / stop worker
# ------------------------------------------------------------------ #


@pytest.mark.anyio
async def test_start_worker_success() -> None:
    sv = _mock_supervisor(_ok(data={"success": True}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers/w1/start")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_start_worker_failure() -> None:
    sv = _mock_supervisor(_ok(data={"success": False}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers/w1/start")
    assert resp.status_code == 500


@pytest.mark.anyio
async def test_stop_worker_success() -> None:
    sv = _mock_supervisor(_ok(data={"success": True}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers/w1/stop")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_stop_worker_failure() -> None:
    sv = _mock_supervisor(_ok(data={"success": False}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.post(f"{PREFIX}/stack/workers/w1/stop")
    assert resp.status_code == 500


# ------------------------------------------------------------------ #
# destroy
# ------------------------------------------------------------------ #


@pytest.mark.anyio
async def test_destroy_worker() -> None:
    sv = _mock_supervisor(_ok(data={"success": True}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.delete(f"{PREFIX}/stack/workers/w1")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_destroy_workers_empty_body() -> None:
    sv = _mock_supervisor(_ok(data={}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.delete(f"{PREFIX}/stack/workers")
    assert resp.status_code == 200
    sv.exec_cmd.assert_awaited_once()
    cmd = sv.exec_cmd.call_args[0][0]
    assert cmd.command.value == "DESTROY_WORKERS"
    assert cmd.payload is None


@pytest.mark.anyio
async def test_destroy_workers_with_names() -> None:
    sv = _mock_supervisor(_ok(data={}))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.request(
            "DELETE",
            f"{PREFIX}/stack/workers",
            content='["w1", "w2"]',
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 200
    cmd = sv.exec_cmd.call_args[0][0]
    assert cmd.payload == {"worker_names": ["w1", "w2"]}


@pytest.mark.anyio
async def test_destroy_workers_invalid_json() -> None:
    sv = _mock_supervisor()
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.request(
            "DELETE",
            f"{PREFIX}/stack/workers",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_destroy_workers_wrong_type() -> None:
    sv = _mock_supervisor()
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.request(
            "DELETE",
            f"{PREFIX}/stack/workers",
            content='{"not": "a list"}',
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 400
    assert "array" in resp.json()["detail"]


# ------------------------------------------------------------------ #
# timeout
# ------------------------------------------------------------------ #


@pytest.mark.anyio
async def test_command_timeout() -> None:
    sv = MagicMock()
    sv.exec_cmd = AsyncMock(side_effect=TimeoutError("timed out"))
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(sv)), base_url="http://t"
    ) as ac:
        resp = await ac.get(f"{PREFIX}/stack/workers")
    assert resp.status_code == 504
