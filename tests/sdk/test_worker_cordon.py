import json

import pytest
import respx
from fastapi.routing import APIRoute
from flowmesh import AsyncFlowMesh, FlowMesh
from flowmesh.models.workers import WorkerCordon, WorkerCordonResult, WorkerInfo
from flowmesh_cli.commands.worker import app
from typer.testing import CliRunner

from server.routers.v1.workers import router as workers_router

from .router_app import TEST_BASE_URL, route_url

_RESULT = {
    "node_alias": "node-a",
    "alias": "fm-worker-0",
    "cordoned": True,
    "changed": True,
}


@pytest.fixture
def mock_client() -> FlowMesh:
    return FlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test-key")


class TestSyncCordon:
    @respx.mock
    def test_cordon(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("cordon_worker", worker_id="wkr-1")).respond(
            json=_RESULT
        )
        assert mock_client.workers.cordon("wkr-1") == WorkerCordonResult.model_validate(
            _RESULT
        )
        assert route.calls[0].request.content == b""

    @respx.mock
    def test_uncordon(self, mock_client: FlowMesh) -> None:
        respx.post(route_url("uncordon_worker", worker_id="wkr-1")).respond(
            json={**_RESULT, "cordoned": False}
        )
        assert mock_client.workers.uncordon("wkr-1").cordoned is False

    @respx.mock
    def test_list_cordons(self, mock_client: FlowMesh) -> None:
        respx.get(route_url("list_cordons")).respond(
            json=[{"node_alias": "node-a", "alias": "alpha"}]
        )
        assert mock_client.workers.list_cordons() == [
            WorkerCordon(node_alias="node-a", alias="alpha")
        ]

    @respx.mock
    def test_remove_cordon(self, mock_client: FlowMesh) -> None:
        route = respx.delete(
            route_url("remove_cordon", node_alias="node-a", alias="alpha")
        ).respond(json={**_RESULT, "alias": "alpha", "cordoned": False})
        assert mock_client.workers.remove_cordon("node-a", "alpha").cordoned is False
        assert route.called

    def test_cordoned_route_precedes_the_worker_id_route(self) -> None:
        assert route_url("get_worker", worker_id="cordoned") == route_url(
            "list_cordons"
        )
        names = [r.name for r in workers_router.routes if isinstance(r, APIRoute)]
        assert names.index("list_cordons") < names.index("get_worker")


class TestAsyncCordon:
    @pytest.mark.asyncio
    @respx.mock
    async def test_async_cordon_and_remove(self) -> None:
        client = AsyncFlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")
        respx.post(route_url("cordon_worker", worker_id="wkr-1")).respond(json=_RESULT)
        respx.delete(
            route_url("remove_cordon", node_alias="node-a", alias="fm-worker-0")
        ).respond(json={**_RESULT, "cordoned": False})
        assert (await client.workers.cordon("wkr-1")).cordoned is True
        result = await client.workers.remove_cordon("node-a", "fm-worker-0")
        assert result.cordoned is False


class TestWorkerModel:
    def test_cordon_fields_default_when_absent(self) -> None:
        info = WorkerInfo.model_validate(
            {
                "id": "wkr-1",
                "namespace": "default",
                "cluster": "default",
                "node_id": "nde-1",
                "node_alias": "node",
                "status": "IDLE",
            }
        )
        assert info.cordoned is False


class TestCLI:
    @respx.mock
    def test_cordon_prints_the_result(self, cli_env: None) -> None:
        respx.post(route_url("cordon_worker", worker_id="wkr-1")).respond(json=_RESULT)
        result = CliRunner().invoke(app, ["cordon", "wkr-1"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["alias"] == "fm-worker-0"

    @respx.mock
    def test_cordon_exits_nonzero_on_conflict(self, cli_env: None) -> None:
        respx.post(route_url("cordon_worker", worker_id="wkr-1")).respond(
            status_code=409, json={"detail": "ambiguous"}
        )
        assert CliRunner().invoke(app, ["cordon", "wkr-1"]).exit_code == 1

    @respx.mock
    def test_cordons_lists_entries(self, cli_env: None) -> None:
        respx.get(route_url("list_cordons")).respond(
            json=[{"node_alias": "node-a", "alias": "alpha"}]
        )
        result = CliRunner().invoke(app, ["cordons"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == [{"node_alias": "node-a", "alias": "alpha"}]

    @respx.mock
    def test_remove_cordon(self, cli_env: None) -> None:
        respx.delete(
            route_url("remove_cordon", node_alias="node-a", alias="alpha")
        ).respond(json={**_RESULT, "alias": "alpha", "cordoned": False})
        result = CliRunner().invoke(app, ["remove-cordon", "node-a", "alpha"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["cordoned"] is False
