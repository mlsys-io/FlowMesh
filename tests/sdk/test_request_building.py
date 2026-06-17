"""Request building tests — verify URL, headers, params via respx mock.

Server-side Pydantic models are used to build mock response payloads so that
any schema change in the server is automatically reflected here.
"""

import json
from typing import Any

import pytest
import respx
from flowmesh import AsyncFlowMesh, FlowMesh
from pydantic import BaseModel

from server.registries.node import Node as SrvNode

# Server-side imports — used only to build realistic JSON response payloads.
from server.registries.workflow import Workflow as SrvWorkflow
from server.registries.workflow import WorkflowStatus as SrvWorkflowStatus
from server.schemas.common import OkResponse as SrvOkResponse
from server.schemas.common import VersionResponse as SrvVersionResponse
from server.schemas.workflow import WorkflowSubmitResponse as SrvWorkflowSubmitResponse
from server.schemas.workflow import (
    WorkflowSubmitTaskEntry as SrvWorkflowSubmitTaskEntry,
)

from .helpers import clear_env_and_config_file
from .router_app import TEST_BASE_URL, route_url


@pytest.fixture
def mock_client() -> FlowMesh:
    return FlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")


# ------------------------------------------------------------------ #
# Shared server model instances
# ------------------------------------------------------------------ #

_WORKFLOW_RESPONSE = SrvWorkflow(
    workflow_id="wf-1",
    task_ids=[],
    submitted_at="2025-01-01T00:00:00Z",
    updated_at="2025-01-01T00:00:00Z",
    status=SrvWorkflowStatus.PENDING,
    dispatched_tasks=[],
    completed_tasks=[],
    failed_tasks=[],
    cancelled_tasks=[],
)

_SUBMIT_RESPONSE = SrvWorkflowSubmitResponse(
    ok=True,
    workflow_id="wf-1",
    count=1,
    tasks=[SrvWorkflowSubmitTaskEntry(task_id="t-1")],
)

_OK_RESPONSE = SrvOkResponse(ok=True)

_VERSION_RESPONSE = SrvVersionResponse(version="0.1.0")

_NODE_RESPONSE = SrvNode(
    id="gdn-1",
    namespace="flowmesh",
    cluster="cluster",
    alias="node-1",
    started_at="2025-01-01T00:00:00Z",
    tags=["gpu", "prod"],
    last_seen="2025-01-01T00:05:00Z",
)


def _json(server_obj: BaseModel) -> dict:  # type: ignore[type-arg]
    """Dump a server Pydantic model to a JSON-compatible dict."""
    return server_obj.model_dump(mode="json")


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestURLConstruction:
    @respx.mock
    def test_get_workflow_url(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_workflow", workflow_id="wf-1")).respond(
            json=_json(_WORKFLOW_RESPONSE)
        )
        mock_client.workflows.retrieve("wf-1")
        assert route.called

    @respx.mock
    def test_list_workers_url(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_workers")).respond(json=[])
        mock_client.workers.list()
        assert route.called

    @respx.mock
    def test_health_url_no_api_prefix(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("healthz")).respond(json=_json(_OK_RESPONSE))
        mock_client.system.health()
        assert route.called

    @respx.mock
    def test_version_url(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_version")).respond(
            json=_json(_VERSION_RESPONSE)
        )
        result = mock_client.system.version()
        assert route.called
        assert result.version == "0.1.0"


class TestAuthHeaders:
    @respx.mock
    def test_no_token_no_auth_header(self) -> None:
        with clear_env_and_config_file():
            client = FlowMesh(base_url=TEST_BASE_URL, api_key=None)
        route = respx.get(route_url("healthz")).respond(json=_json(_OK_RESPONSE))
        client.system.health()
        request = route.calls[0].request
        assert "authorization" not in request.headers


class TestWorkflowSubmit:
    @respx.mock
    def test_native_submit_content_type(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("submit_workflow")).respond(
            json=_json(_SUBMIT_RESPONSE)
        )
        mock_client.workflows.submit("apiVersion: flowmesh/v1\nkind: Task")
        request = route.calls[0].request
        assert request.headers["content-type"] == "text/plain"
        assert request.headers["workflow-format"] == "native"

    @respx.mock
    def test_n8n_submit_content_type(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("submit_workflow")).respond(
            json=_json(_SUBMIT_RESPONSE)
        )
        mock_client.workflows.submit('{"nodes": []}', workflow_format="n8n")
        request = route.calls[0].request
        assert request.headers["content-type"] == "application/json"
        assert request.headers["workflow-format"] == "n8n"

    @respx.mock
    def test_native_submit_dict_serializes_yaml(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("submit_workflow")).respond(
            json=_json(_SUBMIT_RESPONSE)
        )
        mock_client.workflows.submit(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "Workflow",
                "metadata": {"name": "demo"},
                "spec": {"stages": {}},
            }
        )
        request = route.calls[0].request
        assert request.headers["content-type"] == "text/plain"
        assert request.headers["workflow-format"] == "native"
        body = request.content.decode()
        assert "kind: Workflow" in body
        assert "name: demo" in body

    @pytest.mark.anyio
    @respx.mock
    async def test_async_n8n_submit_dict_serializes_json(self) -> None:
        route = respx.post(route_url("submit_workflow")).respond(
            json=_json(_SUBMIT_RESPONSE)
        )
        workflow: dict[str, Any] = {"nodes": [], "connections": {}}
        async with AsyncFlowMesh(
            base_url=TEST_BASE_URL, api_key="flm-test-key"
        ) as client:
            await client.workflows.submit(workflow, workflow_format="n8n")
        request = route.calls[0].request
        assert request.headers["content-type"] == "application/json"
        assert request.headers["workflow-format"] == "n8n"
        assert json.loads(request.content) == workflow

    @respx.mock
    def test_validate_native_dict_serializes_yaml(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("validate_workflow")).respond(
            json={"ok": True, "count": 0, "tasks": []}
        )
        mock_client.workflows.validate(
            {"apiVersion": "flowmesh/v1", "kind": "Workflow", "spec": {"stages": {}}}
        )
        request = route.calls[0].request
        assert request.headers["content-type"] == "text/plain"
        assert "kind: Workflow" in request.content.decode()


class TestQueryParams:
    @respx.mock
    def test_list_nodes_accepts_string_tags(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_nodes")).respond(json=[_json(_NODE_RESPONSE)])
        nodes = mock_client.nodes.list()
        assert route.called
        assert nodes[0].tags == ["gpu", "prod"]

    @respx.mock
    def test_list_tasks_with_filters(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_tasks")).respond(json=[])
        mock_client.tasks.list(workflow_id="wf-1", status=["DONE", "FAILED"])
        request = route.calls[0].request
        url = str(request.url)
        assert "workflow_id=wf-1" in url
        assert "status=DONE" in url
        assert "status=FAILED" in url

    @respx.mock
    def test_list_workers_with_tags(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_workers")).respond(json=[])
        mock_client.workers.list(tags=["gpu", "a100"])
        request = route.calls[0].request
        url = str(request.url)
        assert "tags=gpu" in url
        assert "tags=a100" in url
