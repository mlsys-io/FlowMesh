"""Dispatcher stage-reference resolution tests."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from server.dispatcher.base import Dispatcher
from server.registries.worker import WorkerRegistry
from server.task.models import TaskRecord, TaskStatus
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from shared.schemas.result import APIItem, APIResult, ResultEnvelope
from shared.tasks import TaskEnvelopeTemplate, TaskType
from shared.tasks.specs import ApiSpecTemplate


class _DummyRuntime:
    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        depends_on: dict[str, list[str]] | None = None,
    ) -> None:
        self.tasks = tasks
        self._depends_on = depends_on or {}

    def get_record(self, task_id: str) -> TaskRecord | None:
        return self.tasks.get(task_id)

    def describe_task(self, task_id: str) -> SimpleNamespace | None:
        record = self.tasks.get(task_id)
        if record is None:
            return None
        return SimpleNamespace(depends_on=list(self._depends_on.get(task_id, [])))


def _task_template(task_type: TaskType, **spec_updates: object) -> TaskEnvelopeTemplate:
    payload = {
        "apiVersion": "flowmesh/v1",
        "kind": "Task",
        "metadata": {"name": "wf:task"},
        "spec": {"taskType": task_type.value, **spec_updates},
    }
    return TaskEnvelopeTemplate.model_validate(payload)


def test_api_dependent_stage_resolves_first_row_text(tmp_path: Path) -> None:
    """A dependent stage's ${stage.items.0.text} resolves to the first row's
    text of a batch-only APIResult."""
    stage_dir = tmp_path / "task-api"
    stage_dir.mkdir()
    result = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://api.example.com/v1/chat/completions",
        status_code=200,
        items=[
            APIItem(
                index=0,
                url="https://api.example.com/v1/chat/completions",
                status_code=200,
                text="first row text",
                prompt="first",
            ),
            APIItem(
                index=1,
                url="https://api.example.com/v1/chat/completions",
                status_code=200,
                text="second row text",
                prompt="second",
            ),
        ],
    )
    (stage_dir / "results.json").write_text(
        json.dumps(
            {
                "task_id": "task-api",
                "result": json.loads(
                    ResultEnvelope(task_id="task-api", result=result).model_dump_json()
                )["result"],
            }
        ),
        encoding="utf-8",
    )

    upstream = TaskRecord(
        task_id="task-api",
        workflow_id="wf-1",
        owner_id="owner",
        source="raw",
        task=_task_template(TaskType.API),
        status=TaskStatus.DONE,
        task_type="api",
        local_name="stage",
    )
    dispatcher = Dispatcher(
        runtime=cast(TaskRuntime, _DummyRuntime({})),
        worker_registry=cast(WorkerRegistry, object()),
        results_dir=tmp_path,
        logger=logging.getLogger("test-api-dependent-stage"),
    )

    value = dispatcher._resolve_reference("stage.items.0.text", {"stage": upstream})
    assert value == "first row text"


def test_translated_n8n_dependent_api_stage_resolves(tmp_path: Path) -> None:
    """A translated n8n workflow's dependent API stage resolves end to end."""
    payload = {
        "nodes": [
            {
                "name": "Upstream",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {"values": [{"content": "First answer"}]},
                },
            },
            {
                "name": "Downstream",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {"values": [{"content": "Simplify this"}]},
                },
            },
        ],
        "connections": {
            "Upstream": {"ai_languageModel": [[{"node": "Downstream"}]]},
        },
    }
    parsed = parse_workflow(json.dumps(payload), "n8n")
    by_name = {t.graph_node_name: t for t in parsed.tasks}
    upstream = by_name["Upstream"]
    downstream = by_name["Downstream"]
    assert downstream.depends_on == [upstream.task_id]

    stage_dir = tmp_path / upstream.task_id
    stage_dir.mkdir()
    result = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://api.example.com/v1/chat/completions",
        status_code=200,
        items=[
            APIItem(
                index=0,
                url="https://api.example.com/v1/chat/completions",
                status_code=200,
                text="first row text",
                prompt="first",
            ),
        ],
    )
    (stage_dir / "results.json").write_text(
        json.dumps(
            {
                "task_id": upstream.task_id,
                "result": json.loads(
                    ResultEnvelope(
                        task_id=upstream.task_id, result=result
                    ).model_dump_json()
                )["result"],
            }
        ),
        encoding="utf-8",
    )

    upstream_record = TaskRecord(
        task_id=upstream.task_id,
        workflow_id="wf-1",
        owner_id="owner",
        source="raw",
        task=upstream.task,
        status=TaskStatus.DONE,
        task_type="api",
        graph_node_name="Upstream",
    )
    downstream_record = TaskRecord(
        task_id=downstream.task_id,
        workflow_id="wf-1",
        owner_id="owner",
        source="raw",
        task=downstream.task,
        status=TaskStatus.PENDING,
        task_type="api",
        graph_node_name="Downstream",
    )
    dispatcher = Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {
                    upstream.task_id: upstream_record,
                    downstream.task_id: downstream_record,
                },
                depends_on={downstream.task_id: [upstream.task_id]},
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        results_dir=tmp_path,
        logger=logging.getLogger("test-n8n-dependent-stage"),
    )

    context = dispatcher._build_stage_context(downstream_record)
    spec = cast(ApiSpecTemplate, downstream.task.spec)
    resolved = dispatcher._resolve_placeholders(spec.data, context)
    assert resolved["items"][0] == (
        "The previous stage's response is as follows. Simplify this\n" "first row text"
    )
