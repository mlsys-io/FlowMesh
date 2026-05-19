#!/usr/bin/env python3
"""Data probing utilities for template-driven SQL cost estimation."""

import builtins
import datetime
import logging
import random
from pathlib import Path
from typing import Any

from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import DataProfilingSpecStrict
from shared.utils.json import to_json_serializable, validate_keys

from ..connectors import get_connector_from_spec
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import DataMixin
from .utils.graph_templates import _render_template, _resolve_columns

logger = logging.getLogger(__name__)


class DataProfilingResult(BaseExecutorResult):
    ok: bool = True
    type: str = "sql"
    template: str | None = None
    cost_estimates: dict[str, Any] | None = None


class DataProfilingExecutor(DataMixin, Executor):
    """Executor that estimates SQL query costs by sampling SQL template params."""

    name = "data_profiling"

    def run(self, task: ExecutorTask, out_dir: Path) -> DataProfilingResult:
        spec = self.require_spec(task, DataProfilingSpecStrict)
        task_id = task.task_id
        merge_children = task.merged_children or []

        result = self._run_single_profile(spec, task_id)

        for child in merge_children:
            child_id = child.task_id
            child_spec = child.spec
            if not isinstance(child_spec, DataProfilingSpecStrict):
                raise ExecutionError(
                    "Merged child spec must be data_profiling for merged profiling"
                )
            result.children[child_id] = self._run_single_profile(child_spec, child_id)

        return result

    def _run_single_profile(
        self, spec: DataProfilingSpecStrict, task_id: str
    ) -> DataProfilingResult:
        data_cfg = spec.data
        if not isinstance(data_cfg, dict):
            raise ExecutionError(
                "DataProfilingExecutor.task.spec.data "
                f"for task {task_id} must be a mapping"
            )
        context = spec.upstreamResults or {}

        validate_keys(
            data_cfg,
            f"DataProfilingExecutor.task.spec.data for task {task_id}",
            required={"type", "template", "connection_string"},
        )

        if data_cfg["type"] != "sql":
            raise ExecutionError(
                f"DataProfilingExecutor only supports SQL profiles "
                f"(got: {data_cfg['type']})"
            )

        template: str = data_cfg["template"]
        connection_string: str = data_cfg["connection_string"]
        params: list[dict[str, Any]] = data_cfg.get("params", [])
        constraints: list[dict[str, Any]] = data_cfg.get("constraints", [])
        num_test_queries: int = data_cfg.get("num_test_queries", 1)

        resolved_columns = _resolve_columns(params, context)
        params_cfg: dict[str, dict[str, Any]] = {}
        for col in resolved_columns:
            label: str = col["label"]
            val = col["value"]
            params_cfg[label] = {
                "name": label,
                "candidates": val if isinstance(val, list) else [val],
            }

        for c in constraints:
            name = c["name"]
            params_cfg[name] = c

        cfg: dict[str, Any] = {
            "template": template,
            "params": params_cfg,
        }
        template_str, params_rows, queries = self._sample_template_queries(
            cfg, num_samples=num_test_queries
        )
        cost_estimates = self._estimate_query_costs(
            connection_string, queries, params_rows
        )

        return DataProfilingResult(
            ok=True,
            type="sql",
            template=template_str,
            cost_estimates=cost_estimates,
        )

    def _sample_template_queries(
        self,
        cfg: dict[str, Any],
        num_samples: int,
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        params_cfg: dict[str, dict[str, Any]] = cfg["params"]
        template: str = cfg["template"]

        params_rows = self._sample_template_param_rows(
            params_cfg, num_samples=num_samples
        )

        columns: dict[str, list[Any]] = {
            name: [row[name] for row in params_rows] for name in params_cfg.keys()
        }
        rendered = _render_template(
            columns, template, {name: name for name in params_cfg.keys()}  # type: ignore
        )
        if not all(isinstance(x, str) for x in rendered):
            raise ExecutionError("Rendered template did not produce text output.")
        return template, params_rows, rendered  # type: ignore

    def _sample_template_param_rows(
        self,
        params_cfg: dict[str, dict[str, Any]],
        num_samples: int,
    ) -> list[dict[str, Any]]:
        if num_samples <= 0:
            raise ExecutionError("num_samples must be >= 1 for template sampling")
        rng = random.Random()
        sampled_params: dict[str, list[Any]] = {}
        for param_name, param_cfg in params_cfg.items():
            sampled_params[param_name] = self._sample_template_param_values(
                param_name,
                param_cfg,
                num_samples=num_samples,
                rng=rng,
            )
        _, params_rows = self._normalize_params(sampled_params)
        return params_rows  # type: ignore

    def _sample_template_param_values(
        self,
        param_name: str,
        param_cfg: dict[str, Any],
        num_samples: int,
        rng: random.Random,
    ) -> list[Any]:

        if candidates := param_cfg.get("candidates"):
            candidates = list(set(to_json_serializable(v) for v in candidates))
            return [
                to_json_serializable(rng.choice(candidates)) for _ in range(num_samples)
            ]

        type_spec: str = param_cfg["type"]
        min_val = param_cfg.get("min")
        max_val = param_cfg.get("max")
        if not type_spec:
            raise ExecutionError(
                f"template parameter {param_name} requires "
                "type specification for sampling"
            )

        param_type = self._resolve_param_type(type_spec)
        if param_type is None:
            raise ExecutionError(
                f"template parameter {param_name} has unsupported type {type_spec}"
            )

        match param_type:
            case builtins.bool:
                return [bool(rng.choice([True, False])) for _ in range(num_samples)]
            case builtins.int:
                if min_val is None or max_val is None:
                    raise ExecutionError(
                        f"template parameter {param_name} requires "
                        "min and max for numeric sampling"
                    )
                try:
                    min_val_int = param_type(min_val)
                    max_val_int = param_type(max_val)
                except Exception as exc:
                    raise ExecutionError(
                        f"template parameter {param_name} min/max must be numeric"
                    ) from exc
                if min_val_int > max_val_int:
                    raise ExecutionError(
                        f"template parameter {param_name} min exceeds "
                        f"max ({min_val_int} > {max_val_int})"
                    )
                return [
                    rng.randint(min_val_int, max_val_int) for _ in range(num_samples)
                ]
            case builtins.float:
                if min_val is None or max_val is None:
                    raise ExecutionError(
                        f"template parameter {param_name} requires "
                        "min and max for numeric sampling"
                    )
                try:
                    min_val_float = param_type(min_val)
                    max_val_float = param_type(max_val)
                except Exception as exc:
                    raise ExecutionError(
                        f"template parameter {param_name} min/max must be numeric"
                    ) from exc
                if min_val_float > max_val_float:
                    raise ExecutionError(
                        f"template parameter {param_name} min exceeds "
                        f"max ({min_val_float} > {max_val_float})"
                    )
                return [
                    rng.uniform(min_val_float, max_val_float)
                    for _ in range(num_samples)
                ]
            case datetime.date:
                if min_val is None or max_val is None:
                    raise ExecutionError(
                        f"template parameter {param_name} requires "
                        "min and max for date sampling"
                    )
                min_dt = datetime.date.fromisoformat(str(min_val))
                max_dt = datetime.date.fromisoformat(str(max_val))
                date_delta = (max_dt - min_dt).days
                if date_delta < 0:
                    raise ValueError("min date is after max date")
                return [
                    min_dt + datetime.timedelta(days=rng.randint(0, date_delta))
                    for _ in range(num_samples)  # type: ignore
                ]
            case datetime.datetime:
                if min_val is None or max_val is None:
                    raise ExecutionError(
                        f"template parameter {param_name} requires "
                        "min and max for date sampling"
                    )
                min_dt = datetime.datetime.fromisoformat(str(min_val))
                max_dt = datetime.datetime.fromisoformat(str(max_val))
                datetime_delta = (max_dt - min_dt).total_seconds()
                if datetime_delta < 0:
                    raise ValueError("min datetime is after max datetime")
                return [
                    min_dt + datetime.timedelta(seconds=rng.uniform(0, datetime_delta))
                    for _ in range(num_samples)
                ]
            case _:
                raise ExecutionError(
                    f"template parameter {param_name} has "
                    f"unsupported type {type_spec} for sampling"
                )

    def _estimate_query_costs(
        self,
        connection_string: str,
        queries: list[str],
        params_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        query_costs: list[float] = []
        query_rows: list[int] = []

        with get_connector_from_spec(connection_string) as connector:
            for idx, query in enumerate(queries):
                params_row = (
                    params_rows[idx] if params_rows and idx < len(params_rows) else {}
                )
                cost_info = connector.estimate_query_cost(query)
                if cost_info["error"] is not None:
                    raise Exception(cost_info["error"])
                cost = cost_info["estimated_cost"]
                rows = cost_info["estimated_rows"]

                query_costs.append(float(cost))
                query_rows.append(int(rows))

                results.append(
                    {
                        "query": query,
                        "params": params_row,
                        "estimated_cost": cost,
                        "estimated_rows": rows,
                    }
                )

        avg_cost = sum(query_costs) / len(query_costs)
        min_cost = min(query_costs)
        max_cost = max(query_costs)

        avg_rows = sum(query_rows) / len(query_rows)
        min_rows = min(query_rows)
        max_rows = max(query_rows)

        return {
            "ok": True,
            "num_queries": len(queries),
            "avg_estimated_cost": avg_cost,
            "min_estimated_cost": min_cost,
            "max_estimated_cost": max_cost,
            "avg_estimated_rows": avg_rows,
            "min_estimated_rows": min_rows,
            "max_estimated_rows": max_rows,
        }
