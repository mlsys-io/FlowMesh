"""Request building tests — verify URL, headers, params via respx mock.

Server-side Pydantic models are used to build mock response payloads so that
any schema change in the server is automatically reflected here.
"""

import json
from typing import Any

import httpx
import pytest
import respx
from flowmesh import AsyncFlowMesh, FlowMesh
from flowmesh.models import TaskStatus, WorkflowStatus
from pydantic import BaseModel

from server.registries.node import Node as SrvNode

# Server-side imports — used only to build realistic JSON response payloads.
from server.registries.workflow import Workflow as SrvWorkflow
from server.registries.workflow import WorkflowStatus as SrvWorkflowStatus
from server.schemas.common import OkResponse as SrvOkResponse
from server.schemas.common import VersionResponse as SrvVersionResponse
from server.schemas.task import TaskStatusView as SrvTaskStatusView
from server.schemas.workflow import WorkflowStatusView as SrvWorkflowStatusView
from server.schemas.workflow import WorkflowSubmitResponse as SrvWorkflowSubmitResponse
from server.schemas.workflow import (
    WorkflowSubmitTaskEntry as SrvWorkflowSubmitTaskEntry,
)
from server.task.models import TaskInfo as SrvTaskInfo
from shared.tasks.envelope import TaskEnvelopeTemplate
from shared.tasks.specs.misc import EchoSpecTemplate
from shared.tasks.task_type import TaskType

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


_TASK_RESPONSE = SrvTaskInfo(
    task_id="t-1",
    workflow_id="wf-1",
    owner_id="usr-1",
    source="apiVersion: flowmesh/v1\nkind: Task",
    task=TaskEnvelopeTemplate(
        apiVersion="flowmesh/v1",
        kind="Task",
        spec=EchoSpecTemplate(taskType=TaskType.ECHO),
    ),
    status="DONE",
    submitted_at="2025-01-15T10:30:00Z",
    submitted_ts=1705312200.0,
    depends_on=[],
    pending_dependencies=[],
    dependents=[],
    completed=True,
    failed=False,
)


def _status_json(
    status: str, latest_update: dict[str, Any] | None = None
) -> dict[str, Any]:
    view = SrvTaskStatusView.from_info(_TASK_RESPONSE)
    return _json(
        view.model_copy(update={"status": status, "latest_update": latest_update})
    )


def _workflow_status_json(status: SrvWorkflowStatus) -> dict[str, Any]:
    view = SrvWorkflowStatusView.from_workflow(_WORKFLOW_RESPONSE)
    return _json(view.model_copy(update={"status": status}))


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


class TestTaskPolling:
    @respx.mock
    def test_status_requests_status_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_task", task_id="t-1")).respond(
            json=_status_json("DISPATCHED")
        )
        view = mock_client.tasks.status("t-1")
        assert route.calls[0].request.url.params["view"] == "status"
        assert view.status == "DISPATCHED"

    @respx.mock
    def test_list_sends_limit(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_tasks")).respond(json=[])
        mock_client.tasks.list(workflow_id="wf-1", limit=2)
        params = route.calls[0].request.url.params
        assert params["limit"] == "2"
        assert "view" not in params

    @respx.mock
    def test_list_statuses_requests_status_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_tasks")).respond(
            json=[_status_json("PENDING")]
        )
        views = mock_client.tasks.list_statuses(status="PENDING", limit=5)
        params = route.calls[0].request.url.params
        assert params["view"] == "status"
        assert params["status"] == "PENDING"
        assert params["limit"] == "5"
        assert [v.task_id for v in views] == ["t-1"]

    @respx.mock
    def test_wait_polls_status_then_fetches_full_task(
        self, mock_client: FlowMesh
    ) -> None:
        route = respx.get(route_url("get_task", task_id="t-1")).mock(
            side_effect=[
                httpx.Response(200, json=_status_json("DISPATCHED")),
                httpx.Response(200, json=_status_json("DONE")),
                httpx.Response(200, json=_json(_TASK_RESPONSE)),
            ]
        )
        task = mock_client.tasks.wait("t-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", "status", None]
        assert task.source == _TASK_RESPONSE.source

    @pytest.mark.anyio
    @respx.mock
    async def test_async_wait_polls_status_then_fetches_full_task(self) -> None:
        route = respx.get(route_url("get_task", task_id="t-1")).mock(
            side_effect=[
                httpx.Response(200, json=_status_json("DONE")),
                httpx.Response(200, json=_json(_TASK_RESPONSE)),
            ]
        )
        async with AsyncFlowMesh(
            base_url=TEST_BASE_URL, api_key="flm-test-key"
        ) as client:
            task = await client.tasks.wait("t-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", None]
        assert task.task_id == "t-1"

    @respx.mock
    def test_wait_status_returns_terminal_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_task", task_id="t-1")).mock(
            side_effect=[
                httpx.Response(200, json=_status_json("DISPATCHED")),
                httpx.Response(200, json=_status_json("DONE")),
            ]
        )
        view = mock_client.tasks.wait_status("t-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", "status"]
        assert view.status == TaskStatus.DONE

    @pytest.mark.anyio
    @respx.mock
    async def test_async_wait_status_returns_terminal_view(self) -> None:
        route = respx.get(route_url("get_task", task_id="t-1")).mock(
            side_effect=[
                httpx.Response(200, json=_status_json("DISPATCHED")),
                httpx.Response(200, json=_status_json("FAILED")),
            ]
        )
        async with AsyncFlowMesh(
            base_url=TEST_BASE_URL, api_key="flm-test-key"
        ) as client:
            view = await client.tasks.wait_status("t-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", "status"]
        assert view.status == TaskStatus.FAILED

    @respx.mock
    def test_wait_for_ssh_polls_status_view(self, mock_client: FlowMesh) -> None:
        ssh = {"mode": "direct", "host": "h", "port": 22}
        route = respx.get(route_url("get_task", task_id="t-1")).mock(
            side_effect=[
                httpx.Response(200, json=_status_json("DISPATCHED")),
                httpx.Response(200, json=_status_json("DISPATCHED", {"ssh": ssh})),
            ]
        )
        assert mock_client.tasks.wait_for_ssh("t-1", interval=0) == ssh
        assert [c.request.url.params["view"] for c in route.calls] == ["status"] * 2

    @pytest.mark.anyio
    @respx.mock
    async def test_async_status_calls_request_status_view(self) -> None:
        ssh = {"mode": "direct", "host": "h", "port": 22}
        task_route = respx.get(route_url("get_task", task_id="t-1")).respond(
            json=_status_json("DISPATCHED", {"ssh": ssh})
        )
        list_route = respx.get(route_url("list_tasks")).respond(
            json=[_status_json("DISPATCHED")]
        )
        async with AsyncFlowMesh(
            base_url=TEST_BASE_URL, api_key="flm-test-key"
        ) as client:
            view = await client.tasks.status("t-1")
            info = await client.tasks.wait_for_ssh("t-1", interval=0)
            views = await client.tasks.list_statuses(limit=1)
        assert view.status == "DISPATCHED"
        assert info == ssh
        assert [v.task_id for v in views] == ["t-1"]
        assert all(c.request.url.params["view"] == "status" for c in task_route.calls)
        assert list_route.calls[0].request.url.params["view"] == "status"
        assert list_route.calls[0].request.url.params["limit"] == "1"


class TestWorkflowPolling:
    @respx.mock
    def test_status_requests_status_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_workflow", workflow_id="wf-1")).respond(
            json=_workflow_status_json(SrvWorkflowStatus.DISPATCHED)
        )
        view = mock_client.workflows.status("wf-1")
        assert route.calls[0].request.url.params["view"] == "status"
        assert view.status == WorkflowStatus.DISPATCHED

    @respx.mock
    def test_list_sends_limit(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_workflows")).respond(json=[])
        mock_client.workflows.list(status="DONE", limit=2)
        params = route.calls[0].request.url.params
        assert params["limit"] == "2"
        assert params["status"] == "DONE"
        assert "view" not in params

    @respx.mock
    def test_list_statuses_requests_status_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("list_workflows")).respond(
            json=[_workflow_status_json(SrvWorkflowStatus.PENDING)]
        )
        views = mock_client.workflows.list_statuses(task_ids="t-1", limit=5)
        params = route.calls[0].request.url.params
        assert params["view"] == "status"
        assert params["task_ids"] == "t-1"
        assert params["limit"] == "5"
        assert [v.workflow_id for v in views] == ["wf-1"]

    @respx.mock
    def test_wait_polls_status_then_fetches_full_workflow(
        self, mock_client: FlowMesh
    ) -> None:
        route = respx.get(route_url("get_workflow", workflow_id="wf-1")).mock(
            side_effect=[
                httpx.Response(
                    200, json=_workflow_status_json(SrvWorkflowStatus.DISPATCHED)
                ),
                httpx.Response(200, json=_workflow_status_json(SrvWorkflowStatus.DONE)),
                httpx.Response(200, json=_json(_WORKFLOW_RESPONSE)),
            ]
        )
        workflow = mock_client.workflows.wait("wf-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", "status", None]
        assert workflow.task_ids == []

    @respx.mock
    def test_wait_status_returns_terminal_view(self, mock_client: FlowMesh) -> None:
        route = respx.get(route_url("get_workflow", workflow_id="wf-1")).mock(
            side_effect=[
                httpx.Response(
                    200, json=_workflow_status_json(SrvWorkflowStatus.PENDING)
                ),
                httpx.Response(
                    200, json=_workflow_status_json(SrvWorkflowStatus.FAILED)
                ),
            ]
        )
        view = mock_client.workflows.wait_status("wf-1", interval=0)
        views = [call.request.url.params.get("view") for call in route.calls]
        assert views == ["status", "status"]
        assert view.status == WorkflowStatus.FAILED

    @pytest.mark.anyio
    @respx.mock
    async def test_async_polling_requests_status_view(self) -> None:
        get_route = respx.get(route_url("get_workflow", workflow_id="wf-1")).mock(
            side_effect=[
                httpx.Response(200, json=_workflow_status_json(SrvWorkflowStatus.DONE)),
                httpx.Response(200, json=_workflow_status_json(SrvWorkflowStatus.DONE)),
                httpx.Response(200, json=_json(_WORKFLOW_RESPONSE)),
            ]
        )
        list_route = respx.get(route_url("list_workflows")).respond(
            json=[_workflow_status_json(SrvWorkflowStatus.DONE)]
        )
        async with AsyncFlowMesh(
            base_url=TEST_BASE_URL, api_key="flm-test-key"
        ) as client:
            view = await client.workflows.status("wf-1")
            workflow = await client.workflows.wait("wf-1", interval=0)
            views = await client.workflows.list_statuses(limit=1)
        assert view.status == WorkflowStatus.DONE
        assert workflow.workflow_id == "wf-1"
        assert [v.workflow_id for v in views] == ["wf-1"]
        get_views = [call.request.url.params.get("view") for call in get_route.calls]
        assert get_views == ["status", "status", None]
        assert list_route.calls[0].request.url.params["view"] == "status"
        assert list_route.calls[0].request.url.params["limit"] == "1"


class TestNodeRegister:
    @respx.mock
    def test_register_includes_version_when_given(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("register_node")).respond(
            json={"node_id": "nde-1"}
        )
        mock_client.nodes.register(
            namespace="ns",
            cluster="cl",
            alias="n1",
            started_at="2025-01-01T00:00:00Z",
            version="0.1.0",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["version"] == "0.1.0"

    @respx.mock
    def test_register_omits_version_when_absent(self, mock_client: FlowMesh) -> None:
        route = respx.post(route_url("register_node")).respond(
            json={"node_id": "nde-1"}
        )
        mock_client.nodes.register(
            namespace="ns",
            cluster="cl",
            alias="n1",
            started_at="2025-01-01T00:00:00Z",
        )
        body = json.loads(route.calls[0].request.content)
        assert "version" not in body
