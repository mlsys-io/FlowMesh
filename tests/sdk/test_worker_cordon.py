import json
from typing import Any

import pytest
import respx
from fastapi.routing import APIRoute
from flowmesh import AsyncFlowMesh, FlowMesh
from flowmesh.models.workers import WorkerCordon, WorkerCordonResult, WorkerInfo
from flowmesh_cli.commands.worker import app
from typer.testing import CliRunner

from server.routers.v1.workers import router as workers_router

from .router_app import TEST_BASE_URL, route_url

_RESULT: dict[str, Any] = {
    "node_alias": "node-a",
    "alias": "fm-worker-0",
    "cordoned": True,
    "changed": True,
    "worker_ids": ["wkr-1"],
}
_BY_ID = {"worker_id": "wkr-1"}
_BY_ALIAS = {"node_alias": "node-a", "alias": "fm-worker-0"}


def _sent(route: respx.Route) -> Any:
    return json.loads(route.calls[0].request.content)


@pytest.fixture
def mock_client() -> FlowMesh:
    return FlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test-key")


class TestSyncCordon:
    @respx.mock
    def test_cordon_by_id(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("cordon_worker")).respond(json=_RESULT)
        result = mock_client.workers.cordon("wkr-1")
        assert result == WorkerCordonResult.model_validate(_RESULT)
        assert _sent(route) == _BY_ID

    @respx.mock
    def test_cordon_by_alias(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("cordon_worker")).respond(json=_RESULT)
        mock_client.workers.cordon_alias("node-a", "fm-worker-0")
        assert _sent(route) == _BY_ALIAS

    @respx.mock
    def test_uncordon_by_id(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("uncordon_worker")).respond(
            json={**_RESULT, "cordoned": False}
        )
        assert mock_client.workers.uncordon("wkr-1").cordoned is False
        assert _sent(route) == _BY_ID

    @respx.mock
    def test_uncordon_by_alias(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("uncordon_worker")).respond(
            json={**_RESULT, "cordoned": False, "worker_ids": []}
        )
        result = mock_client.workers.uncordon_alias("node-a", "fm-worker-0")
        assert result.worker_ids == []
        assert _sent(route) == _BY_ALIAS

    @respx.mock
    def test_list_cordons(self, mock_client: FlowMesh) -> None:
        respx.get(route_url("list_cordons")).respond(
            json=[{"node_alias": "node-a", "alias": "alpha"}]
        )
        assert mock_client.workers.list_cordons() == [
            WorkerCordon(node_alias="node-a", alias="alpha")
        ]

    def test_cordons_route_precedes_the_worker_id_route(self) -> None:
        assert route_url("get_worker", worker_id="cordons") == route_url("list_cordons")
        names = [r.name for r in workers_router.routes if isinstance(r, APIRoute)]
        assert names.index("list_cordons") < names.index("get_worker")


class TestAsyncCordon:
    @pytest.mark.asyncio
    @respx.mock
    async def test_async_cordon_by_id_and_uncordon_by_alias(self) -> None:
        client = AsyncFlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")
        cordon = respx.post(route_url("cordon_worker")).respond(json=_RESULT)
        uncordon = respx.post(route_url("uncordon_worker")).respond(
            json={**_RESULT, "cordoned": False}
        )
        assert (await client.workers.cordon("wkr-1")).cordoned is True
        result = await client.workers.uncordon_alias("node-a", "fm-worker-0")
        assert result.cordoned is False
        assert _sent(cordon) == _BY_ID
        assert _sent(uncordon) == _BY_ALIAS


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
    def test_cordon_by_id(self, cli_env: None) -> None:
        route = respx.post(route_url("cordon_worker")).respond(json=_RESULT)
        result = CliRunner().invoke(app, ["cordon", "wkr-1"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["alias"] == "fm-worker-0"
        assert _sent(route) == _BY_ID

    @respx.mock
    def test_uncordon_by_alias(self, cli_env: None) -> None:
        route = respx.post(route_url("uncordon_worker")).respond(
            json={**_RESULT, "cordoned": False}
        )
        result = CliRunner().invoke(
            app, ["uncordon", "--node-alias", "node-a", "--alias", "fm-worker-0"]
        )
        assert result.exit_code == 0, result.output
        assert _sent(route) == _BY_ALIAS

    @pytest.mark.parametrize(
        "args",
        [
            ["cordon"],
            ["cordon", "--alias", "fm-worker-0"],
            ["cordon", "wkr-1", "--node-alias", "node-a", "--alias", "fm-worker-0"],
        ],
    )
    @respx.mock
    def test_cordon_rejects_an_ambiguous_selector(
        self, cli_env: None, args: list[str]
    ) -> None:
        route = respx.post(route_url("cordon_worker")).respond(json=_RESULT)
        assert CliRunner().invoke(app, args).exit_code == 2
        assert not route.called

    @respx.mock
    def test_cordon_exits_nonzero_on_a_server_error(self, cli_env: None) -> None:
        respx.post(route_url("cordon_worker")).respond(
            status_code=404, json={"detail": "worker not found"}
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
