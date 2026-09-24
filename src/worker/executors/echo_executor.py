import logging
from pathlib import Path
from typing import Any

from shared.schemas.result import BaseExecutorResult
from shared.schemas.result import EchoItem as EchoResultItem
from shared.schemas.result import EchoResult
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.task_type import TaskType

from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import DataMixin
from .utils.checkpoints import maybe_upload_traces
from .utils.graph_templates import _evaluate_expr
from .utils.safe_eval import safe_execute_function, safe_materialize_function

logger = logging.getLogger(__name__)

type EchoItem = str | dict[str, str]


class EchoExecutor(DataMixin, Executor):
    name = "echo"
    supported_task_types = frozenset({TaskType.ECHO})

    def _append_outputs(self, out_items: list[EchoResultItem], value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self._append_outputs(out_items, item)
            return
        out_items.append(EchoResultItem(output=value))

    @staticmethod
    def _resolve_expr_item(
        item: dict[str, Any], context: dict[str, BaseExecutorResult]
    ) -> Any:
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

    @staticmethod
    def _resolve_function_arg(
        arg: dict[str, Any], context: dict[str, BaseExecutorResult]
    ) -> Any:
        keys = frozenset(arg)
        if keys == {"items"}:
            items = arg["items"]
            if not isinstance(items, list):
                raise ExecutionError(
                    "echo executor function argument 'items' must be a list"
                )
            return items
        if keys in ({"expr"}, {"node", "path"}):
            return EchoExecutor._resolve_expr_item(arg, context)
        raise ExecutionError(
            "echo executor function argument must have exactly one of "
            f"'items', 'expr', or 'node'+'path'; got keys {sorted(keys)}"
        )

    def _resolve_item(
        self, item: EchoItem, context: dict[str, BaseExecutorResult]
    ) -> Any:
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return self._resolve_expr_item(item, context)
        else:
            raise ExecutionError(
                "echo executor requires each spec.data.items entry to be either "
                "a string literal or a mapping"
            )

    def _run_list(
        self, data_cfg: dict[str, Any], context: dict[str, BaseExecutorResult]
    ) -> list[EchoResultItem]:
        items_cfg = data_cfg.get("items")
        if not isinstance(items_cfg, list):
            raise ExecutionError("echo executor requires spec.data.items to be a list")
        merged_items: list[EchoResultItem] = []
        for item in items_cfg:
            resolved = self._resolve_item(item, context)
            self._append_outputs(merged_items, resolved)
        return merged_items

    def _run_function(
        self,
        data_cfg: dict[str, Any],
        context: dict[str, BaseExecutorResult],
        task_id: str,
    ) -> list[EchoResultItem]:
        fn_code = data_cfg.get("function")
        if not isinstance(fn_code, str) or not fn_code.strip():
            raise ExecutionError(
                f"echo executor task {task_id} requires spec.data.function "
                "to be a non-empty string"
            )
        args_cfg = data_cfg.get("arguments")
        if not isinstance(args_cfg, list):
            raise ExecutionError(
                f"echo executor task {task_id} requires spec.data.arguments "
                "to be a list"
            )
        resolved_args = [self._resolve_function_arg(arg, context) for arg in args_cfg]
        try:
            fn_obj = safe_materialize_function(fn_code)
            output = safe_execute_function(
                fn_obj, tuple(resolved_args), expect_list=True
            )
        except Exception as e:
            raise ExecutionError(
                f"echo executor task {task_id} function failed: {e}"
            ) from e
        return [EchoResultItem(output=element) for element in output]

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
            if not isinstance(context, dict):
                raise ExecutionError(
                    "echo executor requires spec._upstreamResults to be a mapping"
                )

            data_type = data_cfg.get("type")
            if data_type == "function":
                merged_items = self._run_function(data_cfg, context, task_id)
            else:
                merged_items = self._run_list(data_cfg, context)

            result = EchoResult(
                items=merged_items,
                count=len(merged_items),
            )
            deps = self._extract_source_data_ids(spec)
            dependencies_by_task = {task_id: deps}

            self._dump_to_governance(
                task_id=task_id,
                result=result,
                dependencies_by_task=dependencies_by_task,
            )
        maybe_upload_traces(task, out_dir, logger=logger)
        return result
