import logging
from pathlib import Path
from typing import Any

from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import EchoSpecStrict

from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import DataMixin
from .utils.checkpoints import maybe_upload_traces
from .utils.graph_templates import _evaluate_expr

logger = logging.getLogger(__name__)

type EchoItem = str | dict[str, str]


class EchoResult(BaseExecutorResult):
    ok: bool = True
    items: list[dict[str, Any]] = []
    count: int = 0


class EchoExecutor(DataMixin, Executor):
    name = "echo"

    def _append_outputs(self, out_items: list[dict[str, Any]], value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self._append_outputs(out_items, item)
            return
        out_items.append({"output": value})

    @staticmethod
    def _resolve_expr_item(item: dict[str, Any], context: dict[str, Any]) -> Any:
        expr = item.get("expr")
        if not expr:
            node = item.get("node")
            path = item.get("path")
            if node and path:
                expr = f"{node}.{path}"
        if not isinstance(expr, str) or not expr.strip():
            raise ExecutionError(
                "echo executor mapping item must contain either 'expr' or "
                "both 'node' and 'path'"
            )
        resolved = _evaluate_expr(expr.strip(), context)
        if resolved is None:
            raise ExecutionError(
                f"echo executor expression resolved to null: '{expr.strip()}'"
            )
        return resolved

    def _resolve_item(self, item: EchoItem, context: dict[str, Any]) -> Any:
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return self._resolve_expr_item(item, context)
        else:
            raise ExecutionError(
                "echo executor requires each spec.data.items entry to be either "
                "a string literal or a mapping"
            )

    def run(self, task: ExecutorTask, out_dir: Path) -> EchoResult:
        spec = self.require_spec(task, EchoSpecStrict)
        task_id = task.task_id.strip()
        with self._task_span(
            task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            data_cfg = spec.data
            context = spec.upstreamResults or {}

            if not isinstance(data_cfg, dict):
                raise ExecutionError("echo executor requires spec.data to be a mapping")
            items_cfg = data_cfg.get("items")
            if not isinstance(items_cfg, list):
                raise ExecutionError(
                    "echo executor requires spec.data.items to be a list"
                )
            if not isinstance(context, dict):
                raise ExecutionError(
                    "echo executor requires spec._upstreamResults to be a mapping"
                )

            merged_items: list[dict[str, Any]] = []
            for item in items_cfg:
                resolved = self._resolve_item(item, context)
                self._append_outputs(merged_items, resolved)

            result = EchoResult(ok=True, items=merged_items, count=len(merged_items))
            deps = self._extract_source_data_ids(spec)
            dependencies_by_task = {task_id: deps}

            self._dump_to_governance(
                task_id=task_id,
                result=result,
                dependencies_by_task=dependencies_by_task,
            )
        maybe_upload_traces(task, out_dir, logger=logger)
        return result
