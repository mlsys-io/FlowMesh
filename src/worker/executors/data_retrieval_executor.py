#!/usr/bin/env python3
"""Template-driven data retrieval executor for SQL, S3, and lumid-data-app sources.

Lumid retrieval (``type: lumid``) emits one ``DataRetrievalResult`` item per
rendered param row. Every item carries ``run_id``, ``access_chain``, and
``materialized_uri``. Agent-mode items additionally carry ``transcript_url``,
``tokens_in``, ``tokens_out``, ``steps_taken``, and ``replay_latency_ms``.
"""

import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
from PIL import Image

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import (
    BaseExecutorResult,
    DataRetrievalItem,
    DataRetrievalResult,
)
from shared.tasks.specs import DataRetrievalSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.json import validate_keys

from ..connectors import LumidDataConnector, PostgreSQLConnector, S3Connector
from ..utils.serialization import serialize_dataframe
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import DataMixin
from .utils.checkpoints import maybe_upload_artifacts, maybe_upload_traces
from .utils.graph_templates import _render_template, _resolve_columns

logger = logging.getLogger(__name__)


class DataRetrievalExecutor(DataMixin, Executor):
    name = "data_retrieval"
    supported_task_types = frozenset({TaskType.DATA_RETRIEVAL})

    def run(self, task: ExecutorTask, out_dir: Path) -> DataRetrievalResult:
        spec = self.require_spec(task, DataRetrievalSpecStrict)
        task_id = task.task_id
        with self._task_span(
            task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            data_cfg = spec.data
            if not isinstance(data_cfg, dict):
                raise ExecutionError("spec.data must be a mapping for data_retrieval.")
            retrieval_type = data_cfg.get("type")
            if retrieval_type not in {"sql", "s3", "lumid"}:
                raise ExecutionError(
                    "spec.data.type must be 'sql', 's3', or 'lumid' for "
                    "data_retrieval."
                )
            context = spec.upstreamResults or {}

            if retrieval_type == "sql":
                result = self._run_sql(data_cfg, context)
            elif retrieval_type == "s3":
                result = self._run_s3(data_cfg, context, out_dir)
            elif retrieval_type == "lumid":
                result = self._run_lumid(data_cfg, context, out_dir)
            else:
                raise ExecutionError(
                    f"Unsupported data_retrieval type: {retrieval_type!r}."
                )

            deps = self._extract_source_data_ids(spec)
            dependencies_by_task = {task_id: deps}
            self._dump_to_governance(
                task_id=task_id,
                result=result,
                dependencies_by_task=dependencies_by_task,
            )

        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        return result

    def _run_sql(
        self, data_cfg: dict[str, Any], context: dict[str, BaseExecutorResult]
    ) -> DataRetrievalResult:
        """
        Execute SQL queries based on the provided data configuration and context.

        :param data_cfg: Description
        :type data_cfg: dict[str, Any]
        :param context: Description
        :type context: dict[str, BaseExecutorResult]
        :return: Description
        :rtype: dict[str, Any]
        """
        validate_keys(
            data_cfg,
            "DataRetrievalExecutor.spec.data",
            required={"type", "template", "connection_string"},
        )
        template: str = data_cfg["template"]
        params: list[dict[str, Any]] = data_cfg.get("params", [])
        connection_string: str = data_cfg["connection_string"]
        require_non_empty_rows_raw = data_cfg.get("require_non_empty_rows", True)
        if not isinstance(require_non_empty_rows_raw, bool):
            raise ExecutionError("spec.data.require_non_empty_rows must be a boolean.")
        require_non_empty_rows = require_non_empty_rows_raw

        resolved_columns = _resolve_columns(params, context)
        params_dict: dict[str, Any] = {
            col["label"]: col["value"] for col in resolved_columns
        }

        if self._has_grouped_params(params_dict):
            raise ExecutionError("Grouped params are not supported for SQL retrieval.")

        columns_dict, params_rows = self._normalize_params(params_dict)  # type: ignore
        format_kwargs = {key: key for key in params_dict}
        rendered = _render_template(columns_dict, template, format_kwargs)  # type: ignore
        if not all(isinstance(x, str) for x in rendered):
            raise ExecutionError("Rendered SQL template did not produce text output.")
        queries: list[str] = list(rendered)  # type: ignore

        items: list[DataRetrievalItem] = []
        with PostgreSQLConnector(connection_string) as connector:
            for idx, (query, params_row) in enumerate(zip(queries, params_rows)):
                sql_result = connector.execute(query)
                if not sql_result["success"]:
                    raise ExecutionError(
                        f"SQL execution failed for query {idx}: {sql_result['error']}"
                    )
                df: pd.DataFrame = sql_result["data"]
                if df.columns.duplicated().any():
                    logger.warning(
                        "Query %d returned duplicate columns; "
                        "keeping first occurrences.",
                        idx,
                    )
                    df = df.loc[:, ~df.columns.duplicated()]
                if require_non_empty_rows and len(df) == 0:
                    raise ExecutionError(
                        "SQL execution returned zero rows while "
                        "require_non_empty_rows=true "
                        f"(query_index={idx}, query={query!r}, params={params_row!r})"
                    )
                items.append(
                    DataRetrievalItem(
                        index=idx,
                        query=query,
                        params=params_row,
                        table=serialize_dataframe(df),
                        rows=len(df),
                    )
                )

        return DataRetrievalResult(
            items=items,
            count=len(items),
        )

    def _run_s3(
        self,
        data_cfg: dict[str, Any],
        context: dict[str, BaseExecutorResult],
        out_dir: Path,
    ) -> DataRetrievalResult:
        validate_keys(
            data_cfg,
            "DataRetrievalExecutor.spec.data",
            required={
                "type",
                "connection_string",
                "template",
            },
        )
        connection_string: str = data_cfg["connection_string"]
        cert_data: str | None = data_cfg.get("cert_data")

        params_rows: list[dict[str, Any]] | None = None

        template: str = data_cfg["template"]
        params: list[dict[str, Any]] = data_cfg.get("params", [])
        resolved_columns = _resolve_columns(params, context)
        params_dict = {col["label"]: col["value"] for col in resolved_columns}
        param_groups = self._split_param_groups(params_dict)

        encoding = data_cfg.get("encoding", "utf-8")
        if not isinstance(encoding, str):
            raise ExecutionError("spec.data.encoding must be a string.")

        items: list[DataRetrievalItem] = []
        with S3Connector(connection_string, cert_data=cert_data) as connector:
            for group_params in param_groups:
                columns_dict, params_rows = self._normalize_params(group_params)  # type: ignore
                format_kwargs = {key: key for key in group_params}
                rendered = _render_template(columns_dict, template, format_kwargs)  # type: ignore
                if not all(isinstance(x, str) for x in rendered):
                    raise ExecutionError(
                        "Rendered S3 template did not produce text output."
                    )
                group_keys: list[str] = list(rendered)  # type: ignore

                s3_result = connector.execute(group_keys, encoding=encoding)
                if not s3_result["success"]:
                    raise ExecutionError(
                        f"S3 retrieval failed: {s3_result['error']}", retryable=True
                    )
                contents = [
                    self._serialize_s3_content(s3_result["data"][key], out_dir)
                    for key in group_keys
                ]
                item: dict[str, Any] = {
                    "keys": group_keys,
                    "content": contents,
                }
                if params_rows:
                    item["params"] = params_rows
                items.append(DataRetrievalItem.model_validate(item))

        return DataRetrievalResult(
            type="s3",
            items=items,
            metadata=s3_result["metadata"],  # type: ignore
        )

    def _run_lumid(
        self,
        data_cfg: dict[str, Any],
        context: dict[str, BaseExecutorResult],
        out_dir: Path,
    ) -> DataRetrievalResult:
        """Execute a lumid-data-app retrieval in sql, agent, or s3 mode."""
        validate_keys(
            data_cfg,
            "DataRetrievalExecutor.spec.data",
            required={"type", "mode", "lumid_data_url", "lumid_data_token"},
        )
        mode: str = data_cfg["mode"]
        if mode not in {"sql", "agent", "s3"}:
            raise ExecutionError(
                "spec.data.mode must be 'sql', 'agent', or 's3' for "
                "data_retrieval(lumid)."
            )
        base_url: str = data_cfg["lumid_data_url"]
        token: str = data_cfg["lumid_data_token"]
        verify_raw = data_cfg.get("verify", False)
        if not isinstance(verify_raw, bool):
            raise ExecutionError("spec.data.verify must be a boolean.")
        output_format: str = data_cfg.get("output_format", "jsonl")
        if mode in {"sql", "agent"} and output_format not in {"jsonl", "csv"}:
            raise ExecutionError(
                "spec.data.output_format must be 'jsonl' or 'csv' for "
                f"data_retrieval(lumid, mode={mode})."
            )

        artifact_dir = out_dir / "artifacts" / "lumid_retrievals"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        items: list[DataRetrievalItem] = []
        with LumidDataConnector(
            base_url=base_url, token=token, verify=verify_raw
        ) as conn:
            if mode == "sql":
                validate_keys(
                    data_cfg,
                    "DataRetrievalExecutor.spec.data",
                    required={"type", "mode", "lumid_data_url", "template"},
                )
                template: str = data_cfg["template"]
                params: list[dict[str, Any]] = data_cfg.get("params", [])
                require_non_empty_rows_raw = data_cfg.get(
                    "require_non_empty_rows", True
                )
                if not isinstance(require_non_empty_rows_raw, bool):
                    raise ExecutionError(
                        "spec.data.require_non_empty_rows must be a boolean.",
                        retryable=True,
                    )
                require_non_empty_rows = require_non_empty_rows_raw

                resolved_columns = _resolve_columns(params, context)
                params_dict: dict[str, Any] = {
                    col["label"]: col["value"] for col in resolved_columns
                }
                if self._has_grouped_params(params_dict):
                    raise ExecutionError(
                        "Grouped params are not supported for lumid sql retrieval."
                    )
                columns_dict, params_rows = self._normalize_params(params_dict)
                format_kwargs = {key: key for key in params_dict}
                rendered = _render_template(
                    cast("dict[str, Sequence[Any]]", columns_dict),
                    template,
                    format_kwargs,
                )
                queries = [x for x in rendered if isinstance(x, str)]
                if len(queries) != len(rendered):
                    raise ExecutionError(
                        "Rendered SQL template did not produce text output."
                    )

                for idx, (query, params_row) in enumerate(zip(queries, params_rows)):
                    out_path = artifact_dir / f"result_{idx}.{output_format}"
                    outcome = conn.execute(
                        query,
                        mode="sql",
                        out_path=out_path,
                        output_format=output_format,
                    )
                    if not outcome["success"]:
                        raise ExecutionError(
                            f"lumid sql retrieval failed (idx={idx}): "
                            f"{outcome['error']}"
                        )
                    df = self._load_table(out_path, output_format)
                    if require_non_empty_rows and len(df) == 0:
                        raise ExecutionError(
                            "lumid sql retrieval returned zero rows while "
                            "require_non_empty_rows=true "
                            f"(query_index={idx}, query={query!r}, "
                            f"params={params_row!r})"
                        )
                    items.append(
                        DataRetrievalItem(
                            index=idx,
                            query=query,
                            params=params_row,
                            table=serialize_dataframe(df),
                            rows=len(df),
                            run_id=outcome["data"]["run_id"],
                            access_chain=outcome["data"]["access_chain"],
                            materialized_uri=outcome["data"]["materialized_uri"],
                            size_bytes=outcome["data"]["size_bytes"],
                        )
                    )

            elif mode == "agent":
                validate_keys(
                    data_cfg,
                    "DataRetrievalExecutor.spec.data",
                    required={"type", "mode", "lumid_data_url", "description"},
                )
                description_template: str = data_cfg["description"]
                schema_scope: str | None = data_cfg.get("schema_scope")
                max_steps_raw = data_cfg.get("max_steps")
                max_steps: int | None = None
                if max_steps_raw is not None:
                    if not isinstance(max_steps_raw, int):
                        raise ExecutionError("spec.data.max_steps must be an integer.")
                    max_steps = max_steps_raw
                model: str | None = data_cfg.get("model")

                params_agent: list[dict[str, Any]] = data_cfg.get("params", [])
                resolved_columns_agent = _resolve_columns(params_agent, context)
                params_dict_agent: dict[str, Any] = {
                    col["label"]: col["value"] for col in resolved_columns_agent
                }
                if self._has_grouped_params(params_dict_agent):
                    raise ExecutionError(
                        "Grouped params are not supported for lumid agent retrieval."
                    )
                columns_dict_agent, params_rows_agent = self._normalize_params(
                    params_dict_agent
                )
                format_kwargs_agent = {key: key for key in params_dict_agent}
                rendered_agent = _render_template(
                    cast("dict[str, Sequence[Any]]", columns_dict_agent),
                    description_template,
                    format_kwargs_agent,
                )
                descriptions: list[str] = []
                for value in rendered_agent:
                    if not isinstance(value, str):
                        raise ExecutionError(
                            "Rendered description template did not produce text output."
                        )
                    descriptions.append(value)

                for idx, (description, params_row) in enumerate(
                    zip(descriptions, params_rows_agent)
                ):
                    out_path = artifact_dir / f"result_{idx}.{output_format}"
                    outcome = conn.execute(
                        description,
                        mode="agent",
                        out_path=out_path,
                        output_format=output_format,
                        schema_scope=schema_scope,
                        max_steps=max_steps,
                        model=model,
                    )
                    if not outcome["success"]:
                        raise ExecutionError(
                            f"lumid agent retrieval failed (idx={idx}): "
                            f"{outcome['error']}"
                        )
                    df = self._load_table(out_path, output_format)
                    outcome_data = outcome["data"]
                    items.append(
                        DataRetrievalItem(
                            index=idx,
                            description=description,
                            params=params_row,
                            table=serialize_dataframe(df),
                            rows=len(df),
                            access_chain=outcome_data["access_chain"],
                            run_id=outcome_data["run_id"],
                            transcript_url=outcome_data["transcript_url"],
                            tokens_in=outcome_data["tokens_in"],
                            tokens_out=outcome_data["tokens_out"],
                            steps_taken=outcome_data["steps_taken"],
                            replay_latency_ms=outcome_data["replay_latency_ms"],
                            materialized_uri=outcome_data["materialized_uri"],
                        )
                    )

            else:
                validate_keys(
                    data_cfg,
                    "DataRetrievalExecutor.spec.data",
                    required={"type", "mode", "lumid_data_url", "template"},
                )
                encoding: str = data_cfg.get("encoding", "utf-8")
                if not isinstance(encoding, str):
                    raise ExecutionError("spec.data.encoding must be a string.")
                template_s3: str = data_cfg["template"]
                params_s3: list[dict[str, Any]] = data_cfg.get("params", [])
                resolved_columns_s3 = _resolve_columns(params_s3, context)
                params_dict_s3 = {
                    col["label"]: col["value"] for col in resolved_columns_s3
                }
                param_groups = self._split_param_groups(params_dict_s3)

                for group_params in param_groups:
                    columns_dict_s3, params_rows_s3 = self._normalize_params(
                        group_params
                    )
                    format_kwargs_s3 = {key: key for key in group_params}
                    rendered_s3 = _render_template(
                        cast("dict[str, Sequence[Any]]", columns_dict_s3),
                        template_s3,
                        format_kwargs_s3,
                    )
                    group_keys = [x for x in rendered_s3 if isinstance(x, str)]
                    if len(group_keys) != len(rendered_s3):
                        raise ExecutionError(
                            "Rendered object template did not produce text output."
                        )

                    obj_result = conn.execute(group_keys, mode="s3", encoding=encoding)
                    if not obj_result["success"]:
                        raise ExecutionError(
                            f"lumid s3 retrieval failed: {obj_result['error']}"
                        )
                    obj_data = obj_result["data"]
                    contents = [
                        self._serialize_s3_content(obj_data[key], out_dir)
                        for key in group_keys
                    ]
                    item: dict[str, Any] = {
                        "keys": group_keys,
                        "content": contents,
                    }
                    if params_rows_s3:
                        item["params"] = params_rows_s3
                    items.append(DataRetrievalItem.model_validate(item))

        return DataRetrievalResult(
            type="lumid",
            items=items,
            count=len(items),
        )

    def _load_table(self, path: Path, output_format: str) -> pd.DataFrame:
        """Load the materialized retrieval file into a DataFrame."""
        if path.stat().st_size == 0:
            return pd.DataFrame()
        if output_format == "jsonl":
            return pd.read_json(path, lines=True)
        return pd.read_csv(path)

    def _serialize_s3_content(self, content: Any, out_dir: Path) -> Any:
        if isinstance(content, pd.DataFrame):
            return serialize_dataframe(content)
        if isinstance(content, Image.Image):
            images_dir = out_dir / "artifacts" / "s3_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid.uuid4().hex}.png"
            file_path = images_dir / filename
            content.save(file_path, format="PNG")
            return ArtifactRef(path=f"s3_images/{filename}")
        return content
