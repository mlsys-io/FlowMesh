"""Worker mixin: data prep helpers (prompts, images, params, dataset shards)."""

import copy
import datetime
import io
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from PIL import Image

from shared.tasks.specs import TaskSpecStrictBase
from shared.utils.json import safe_get

from ...connectors import get_connector_from_spec
from ..base_executor import ExecutionError
from ..utils.artifacts import maybe_resolve_artifact_ref, resolve_artifact
from ..utils.data_utils import normalize_prompt_payload
from ..utils.graph_templates import (
    _evaluate_expr,
    _resolve_columns,
    build_prompts_from_graph_template,
)
from .governance import GovernanceMixin

logger = logging.getLogger(__name__)

type PromptMessage = dict[str, str]
type PromptInput = str | Sequence[PromptMessage]


@dataclass(slots=True)
class InferenceEntry:
    task_id: str
    prompts: list[PromptInput]
    inference_cfg: dict[str, Any]
    data_cfg: dict[str, Any]
    metadata_raw: list[Any]
    append_system_prompt: bool
    images: list[Image.Image | None]
    image_group_sizes: list[int] | None
    image_embedding_path: Path | None
    tables: list[pd.DataFrame]


class DataMixin(GovernanceMixin):
    """Data prep helpers (prompts, images, params, dataset shards).

    Inherits :class:`GovernanceMixin` so every data-prep executor also gets
    the trace + lineage emission surface (``_task_span`` / ``_span`` /
    ``_log_event`` / ``_record_output`` / ``_dump_to_governance``).
    """

    _TEMPLATE_TYPE_MAP: dict[str, type] = {
        "str": str,
        "string": str,
        "text": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "double": float,
        "decimal": float,
        "bool": bool,
        "boolean": bool,
        "date": datetime.date,
        "datetime": datetime.datetime,
        "timestamp": datetime.datetime,
    }

    @classmethod
    def _resolve_param_type(cls, type_spec: str) -> type | None:
        normalized = type_spec.strip().lower()
        return cls._TEMPLATE_TYPE_MAP.get(normalized)

    def _load_image_from_artifact(self, source: str) -> Image.Image:
        """Resolve an artifact URL or local path and load as an RGB PIL image."""
        local_path = resolve_artifact(source)
        try:
            with Image.open(local_path) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise ExecutionError(
                "When fetch_images is True, artifact items must "
                "resolve to valid image files."
            ) from exc
        finally:
            try:
                local_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temporary artifact file: %s", local_path)

    def _normalize_s3_cfg(self, s3_cfg: Any) -> tuple[str, str | None, str]:
        if not isinstance(s3_cfg, dict):
            raise ExecutionError("s3_cfg must be an object")
        connection_string = s3_cfg.get("connection_string")
        if not isinstance(connection_string, str) or not connection_string:
            raise ExecutionError(
                "s3_cfg.connection_string is required and must be a string"
            )
        cert_data = s3_cfg.get("cert_data")
        if cert_data is not None:
            if not isinstance(cert_data, str):
                raise ExecutionError("s3_cfg.cert_data must be a string")
            if cert_data and "BEGIN CERTIFICATE" not in cert_data:
                raise ExecutionError(
                    "s3_cfg.cert_data does not look like a PEM certificate"
                )
        encoding = s3_cfg.get("encoding", "utf-8")
        if not isinstance(encoding, str):
            raise ExecutionError("s3_cfg.encoding must be a string")
        return connection_string, cert_data, encoding

    def _s3_key_from_url(self, url: str, connection_string: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            raise ExecutionError("S3 URL must start with s3://")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ExecutionError("S3 URL must include bucket and object key")

        conn_parsed = urlparse(connection_string)
        if conn_parsed.scheme != "s3":
            raise ExecutionError("s3_cfg.connection_string must start with s3://")
        conn_bucket = conn_parsed.path.lstrip("/").split("/", 1)[0]
        if not conn_bucket:
            raise ExecutionError("s3_cfg.connection_string must include bucket")
        if bucket != conn_bucket:
            raise ExecutionError(
                "S3 URL bucket does not match s3_cfg.connection_string bucket"
            )
        return key

    def _flatten_grouped_image_items(
        self, items: list[Any]
    ) -> tuple[list[Any], list[int] | None]:
        if not items:
            return items, None

        top_level_is_list = [isinstance(item, list) for item in items]
        if all(top_level_is_list):
            flattened: list[list] = []
            group_sizes: list[int] = []
            for group_idx, group in enumerate(items):
                assert isinstance(group, list)
                if not group:
                    raise ExecutionError(
                        "When fetch_images is True, grouped image items "
                        "must not contain "
                        f"empty groups (group index={group_idx})."
                    )
                flattened.extend(group)
                group_sizes.append(len(group))
            return flattened, group_sizes

        if any(top_level_is_list):
            raise ExecutionError(
                "When fetch_images is True, list data must be either a flat list or "
                "a grouped list (list[list[...]]). Mixed top-level structures are not "
                "supported."
            )

        return items, None

    @staticmethod
    def _validate_image_group_sizes(
        raw_group_sizes: Any, *, task_id: str | None = None
    ) -> list[int]:
        if (
            not isinstance(raw_group_sizes, list)
            or not raw_group_sizes
            or not all(isinstance(size, int) and size > 0 for size in raw_group_sizes)
        ):
            suffix = f" (task={task_id})." if task_id else "."
            raise ExecutionError(
                "image_group_sizes must be a non-empty list of positive integers"
                f"{suffix}"
            )
        return list(raw_group_sizes)

    @staticmethod
    def _build_group_ranges(
        group_sizes: list[int],
        *,
        total_count: int,
        task_id: str | None = None,
        value_name: str = "items",
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for group_size in group_sizes:
            start = cursor
            end = start + group_size
            ranges.append((start, end))
            cursor = end
        if cursor != total_count:
            suffix = f" (task={task_id})." if task_id else "."
            raise ExecutionError(
                f"{value_name} count mismatch against image_group_sizes: "
                f"sum(group_sizes)={cursor} count={total_count}{suffix}"
            )
        return ranges

    @staticmethod
    def _has_grouped_params(params: dict[str, Any]) -> bool:
        for value in params.values():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(v, list) for v in value)
            ):
                return True
        return False

    @staticmethod
    def _split_param_groups(params: dict[str, Any]) -> list[dict[str, Any]]:
        group_count: int | None = None
        for value in params.values():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(v, list) for v in value)
            ):
                if group_count is None:
                    group_count = len(value)
                elif len(value) != group_count:
                    raise ExecutionError(
                        "Grouped params must have the same number of groups."
                    )

        if group_count is None:
            return [params]

        groups: list[dict[str, Any]] = []
        for idx in range(group_count):
            group_params: dict[str, Any] = {}
            for key, value in params.items():
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, list) for v in value)
                ):
                    group_params[key] = value[idx]
                elif isinstance(value, list) and len(value) > 1:
                    if len(value) == group_count:
                        group_params[key] = value[idx]
                    else:
                        raise ExecutionError(
                            "Grouped params must align across parameters."
                        )
                else:
                    group_params[key] = value
            groups.append(group_params)
        return groups

    def _normalize_params(
        self, params: dict[str, Any]
    ) -> tuple[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]]:
        if not params:
            return {}, [{}]

        columns: dict[str, list[Any]] = {}
        list_lengths: list[int] = []
        for key, value in params.items():
            if isinstance(value, list):
                if not value:
                    raise ExecutionError(f"spec.data.params.{key} must not be empty.")
                columns[key] = value
                if len(value) > 1:
                    list_lengths.append(len(value))
            else:
                columns[key] = [value]

        num_rows = list_lengths[0] if list_lengths else 1
        if list_lengths and any(length != num_rows for length in list_lengths):
            raise ExecutionError("All list-type params must have the same length.")

        params_rows: list[dict[str, Any]] = []
        for idx in range(num_rows):
            params_rows.append(
                {
                    key: (values[idx] if len(values) > 1 else values[0])
                    for key, values in columns.items()
                }
            )
        return columns, params_rows

    @staticmethod
    def _spec_data_cfg(spec: TaskSpecStrictBase) -> dict[str, Any]:
        data = getattr(spec, "data", None) or {}
        if not isinstance(data, dict):
            raise ExecutionError("spec.data must be a mapping.")
        return data

    @staticmethod
    def _spec_inference_cfg(spec: TaskSpecStrictBase) -> dict[str, Any]:
        inference = getattr(spec, "inference", None) or {}
        if not isinstance(inference, dict):
            raise ExecutionError("spec.inference must be a mapping.")
        return inference

    def _collect_prompts_for_spec(
        self, spec: TaskSpecStrictBase, task_id: str, fetch_images: bool = False
    ) -> InferenceEntry:
        data = self._spec_data_cfg(spec)
        if not data:
            raise ExecutionError("spec.data is required.")

        inference_cfg = copy.deepcopy(self._spec_inference_cfg(spec))

        dtype = data.get("type")
        append_system_prompt = True
        prompts: list[PromptInput] = []
        table_stores_list: list[pd.DataFrame] = []
        images: list[Image.Image | None] = []
        image_group_sizes: list[int] | None = None
        metadata_raw: list[Any] = []
        s3_connection_string: str | None = None
        s3_cert_data: str | None = None
        s3_encoding: str | None = None
        if s3_cfg := data.get("s3_cfg"):
            s3_connection_string, s3_cert_data, s3_encoding = self._normalize_s3_cfg(
                s3_cfg
            )
        if dtype == "dataset":
            data_url = data.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'.")
            name = data.get("name", None)
            split = data.get("split", "train")
            shuffle = bool(data.get("shuffle", False))
            trust_remote_code = data.get("trust_remote_code")
            revision = data.get("revision")
            dataset_kwargs = {
                "name": name,
                "split": split,
                "revision": revision,
            }
            if trust_remote_code is not None:
                dataset_kwargs["trust_remote_code"] = bool(trust_remote_code)
            dataset = load_dataset(
                data_url,
                **{k: v for k, v in dataset_kwargs.items() if v is not None},
            )
            dataset = cast(Dataset, dataset)
            if shuffle:
                seed = int(data.get("seed", 42))
                buffer_size = data.get("buffer_size", None)
                dataset = (
                    dataset.shuffle(seed=seed)
                    if buffer_size is None
                    else dataset.shuffle(
                        seed=seed,
                        buffer_size=int(buffer_size),  # type: ignore[arg-type]
                    )
                )
            dataset = self._maybe_apply_dataset_shard(dataset, spec)
            column = data.get("column", "text")
            if column not in dataset.column_names:
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. "
                    f"Available: {dataset.column_names}"
                )
            prompts = [str(x) for x in dataset[column]]

            metadata_spec = data.get("metadata_columns")
            include_all = bool(data.get("metadata_include_all", False))
            rename_map: dict[str, str] = {}
            metadata_keys: list[str] = []
            if isinstance(metadata_spec, dict):
                rename_map = {str(k): str(v) for k, v in metadata_spec.items()}
                metadata_keys = list(rename_map.keys())
            elif isinstance(metadata_spec, list):
                metadata_keys = [str(x) for x in metadata_spec]
            elif metadata_spec is None:
                metadata_keys = []
            else:
                raise ExecutionError(
                    "spec.data.metadata_columns must be a list or mapping"
                )

            for idx in range(len(dataset)):
                row = dataset[idx]
                entry_meta: dict[str, Any] = {"prompt": row.get(column)}
                if metadata_keys:
                    for key in metadata_keys:
                        if key not in row:
                            continue
                        alias = rename_map.get(key, key)
                        entry_meta[alias] = row[key]
                elif include_all:
                    for key, value in row.items():
                        if key == column:
                            continue
                        entry_meta[key] = value
                metadata_raw.append(entry_meta)
        elif dtype == "list":
            items = data.get("items")
            if items is None:
                expr = data.get("expr")
                if not expr:
                    node = data.get("node")
                    path = data.get("path")
                    if node and path:
                        expr = f"{node}.{path}"
                if expr:
                    context = self._spec_upstream_results(spec)
                    resolved_expr = expr.strip()
                    items = _evaluate_expr(resolved_expr, context)
                    root_node = resolved_expr.split(".", 1)[0] or None
                    if isinstance(items, list):
                        items = [
                            maybe_resolve_artifact_ref(item, context, root_node)
                            for item in items
                        ]
            if not isinstance(items, list):
                raise ExecutionError(
                    "spec.data.items must be a list or resolve to a list "
                    "for type == 'list'."
                )
            if fetch_images:
                items, image_group_sizes = self._flatten_grouped_image_items(items)

                s3_entries: list[tuple[int, str]] = []

                for idx, item in enumerate(items):
                    if isinstance(item, Image.Image):
                        images.append(item)
                    elif isinstance(item, str):
                        parsed = urlparse(item)
                        if parsed.scheme in ("", "file"):
                            images.append(self._load_image_from_artifact(item))
                            continue
                        if parsed.scheme not in ("s3", "http", "https"):
                            raise ExecutionError(
                                "When fetch_images is True, all string items must be "
                                "valid HTTP/HTTPS/S3 URLs or local filesystem paths."
                            )
                        if parsed.scheme == "s3":
                            s3_entries.append((idx, item))
                            images.append(None)
                            continue
                        response = requests.get(item, timeout=15)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                        images.append(image)
                    elif isinstance(item, dict):
                        artifact_url = item.get("url")
                        if not isinstance(artifact_url, str) or not artifact_url:
                            raise ExecutionError(
                                "When fetch_images is True, dict items must "
                                "include a string 'url' field. Use a placeholder "
                                "expression (e.g. ${stage.images}) so "
                                "artifact refs are substituted to URL strings."
                            )
                        images.append(self._load_image_from_artifact(artifact_url))
                    else:
                        raise ExecutionError(
                            "When fetch_images is True, items must be Image.Image, "
                            "artifact specs, or HTTP/HTTPS/S3 URLs."
                        )

                if s3_entries:
                    if not s3_connection_string:
                        raise ExecutionError(
                            "spec.data.s3_cfg is required for s3:// items in list data"
                        )
                    keys = [
                        self._s3_key_from_url(url, s3_connection_string)
                        for _, url in s3_entries
                    ]
                    with get_connector_from_spec(
                        s3_connection_string, cert_data=s3_cert_data
                    ) as s3_connector:
                        s3_result = s3_connector.execute(
                            query=keys, encoding=s3_encoding
                        )
                    if not s3_result.get("success"):
                        raise ExecutionError(
                            f"S3 retrieval failed: {s3_result.get('error')}"
                        )
                    file_data = s3_result.get("data")
                    if file_data is None:
                        raise ExecutionError("S3 returned no data for list retrieval")
                    data_dict = file_data.to_dict()
                    for (idx, _), key in zip(s3_entries, keys):
                        content = data_dict.get(key)
                        if not isinstance(content, Image.Image):
                            raise ExecutionError(
                                "S3 object did not resolve to an image. "
                                "Ensure the object is a supported image type."
                            )
                        images[idx] = content.convert("RGB")

                if any(x is None for x in images):
                    raise ExecutionError("Missing image data for one or more items.")
                prompts = [x if isinstance(x, str) else "" for x in items]
            else:
                prompts, apply_chat_template, found_system_prompt = (
                    normalize_prompt_payload(items)
                )
                if apply_chat_template:
                    inference_cfg["apply_chat_template"] = True
                if found_system_prompt:
                    append_system_prompt = False
            raw_meta = data.get("metadata") or data.get("items_metadata") or []
            if raw_meta:
                if not isinstance(raw_meta, list):
                    raise ExecutionError(
                        "spec.data.metadata must be a list when type == 'list'."
                    )
                metadata_raw = list(raw_meta)
        elif dtype == "graph_template":
            upstream_results = self._spec_upstream_results(spec)
            logger.debug(
                "Task %s graph_template upstream keys: %s",
                task_id,
                list(upstream_results.keys()),
            )
            with self._span("build prompt from graph template", data_id=task_id):
                prompts = build_prompts_from_graph_template(data, spec)
            template_cfg = data.get("template") or {}
            append_system_prompt = bool(template_cfg.get("append_system_prompt", False))
        elif dtype == "dataframe":
            upstream_results = self._spec_upstream_results(spec)
            logger.debug(
                "Task %s dataframe upstream keys: %s",
                task_id,
                list(upstream_results.keys()),
            )
            df_columns_cfg = data.get("columns")
            if df_columns_cfg is None:
                raise ExecutionError(
                    "spec.data.columns is required for type == 'dataframe'."
                )
            messages = data.get("messages")
            if not messages:
                raise ExecutionError(
                    "spec.data.messages is required for type == 'dataframe'."
                )
            if not isinstance(messages, list):
                raise ExecutionError(
                    "spec.data.messages must be a list for type == 'dataframe'."
                )
            resolved_columns = _resolve_columns(df_columns_cfg, upstream_results)
            if not resolved_columns:
                raise ExecutionError(
                    "spec.data.columns must resolve to at least one column "
                    "for type == 'dataframe'."
                )

            grouped_columns: dict[str, list[list[Any]]] = {}
            for column in resolved_columns:
                label = column["label"]
                value = column["value"]
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, list) for v in value)
                ):
                    groups = value
                elif isinstance(value, list):
                    groups = [value]
                else:
                    groups = [[value]]
                grouped_columns[label] = groups

            group_count = max(len(groups) for groups in grouped_columns.values())
            for label, groups in list(grouped_columns.items()):
                if len(groups) == 1 and group_count > 1:
                    grouped_columns[label] = groups * group_count
                elif len(groups) != group_count:
                    raise ExecutionError(
                        "spec.data.columns must resolve to the same number of groups."
                    )

            table_stores_list = []
            for group_idx in range(group_count):
                max_len = 1
                raw_group_values: dict[str, list[Any]] = {}
                for label, groups in grouped_columns.items():
                    values = groups[group_idx]
                    if not isinstance(values, list):
                        values = [values]
                    if values:
                        max_len = max(max_len, len(values))
                    raw_group_values[label] = values

                normalized_rows: dict[str, list[Any]] = {}
                for label, values in raw_group_values.items():
                    if len(values) == 1 and max_len > 1:
                        values = [values[0] for _ in range(max_len)]
                    elif len(values) != max_len:
                        raise ExecutionError(
                            "spec.data.columns must resolve to "
                            "the same number of rows per group."
                        )
                    normalized_rows[label] = values

                df = pd.DataFrame(normalized_rows)
                table_stores_list.append(df)

                if fetch_images:
                    contents = df.get(
                        "content", pd.Series(["" for _ in range(len(df))])
                    ).to_list()
                    if not all(isinstance(x, Image.Image) for x in contents):
                        raise ExecutionError(
                            "spec.data.columns must resolve to Image.Image values "
                            "when fetch_images is True."
                        )
                    images.extend(contents)  # type: ignore
                    prompts.extend(["" for _ in contents])
                else:
                    for row in df.to_dict(orient="records"):
                        prompts.append(
                            [
                                {
                                    "role": msg.get("role", "user"),
                                    "content": msg["content"].format(
                                        **row,  # type: ignore[arg-type]
                                    ),
                                }
                                for msg in messages
                            ]
                        )
        else:
            raise ExecutionError(f"Unsupported spec.data.type: {dtype!r}")

        image_embedding: Path | None = None
        if image_embedding_raw := data.get("image_embedding"):
            context = self._spec_upstream_results(spec)
            expr = image_embedding_raw.get("expr")
            node_hint: str | None = image_embedding_raw.get("node")
            if not expr:
                path_hint = image_embedding_raw.get("path")
                if node_hint and path_hint:
                    expr = f"{node_hint}.{path_hint}"
            if not expr:
                raise ExecutionError(
                    "spec.data.image_embedding requires "
                    "'expr' or both 'node' and 'path'"
                )
            resolved_node = node_hint
            if not resolved_node and isinstance(expr, str):
                resolved_node = expr.split(".", 1)[0].strip() or None
            image_embedding_spec: Any = _evaluate_expr(expr.strip(), context)
            artifact_source = maybe_resolve_artifact_ref(
                image_embedding_spec, context, resolved_node
            )
            if not isinstance(artifact_source, str) or not artifact_source:
                raise ExecutionError(
                    "spec.data.image_embedding must resolve to an artifact ref "
                    "({path: ...}) or a URL/path string"
                )

            logger.info("Resolving image embedding from artifact: %s", artifact_source)
            image_embedding = resolve_artifact(artifact_source)
            source_node = image_embedding_raw.get("node")
            if not source_node and isinstance(expr, str) and expr:
                source_node = expr.split(".", 1)[0]
            if isinstance(source_node, str) and source_node:
                upstream_group_sizes = safe_get(
                    context, f"{source_node}.image_group_sizes"
                )
                if upstream_group_sizes is not None:
                    try:
                        image_group_sizes = self._validate_image_group_sizes(
                            upstream_group_sizes,
                            task_id=task_id,
                        )
                    except ExecutionError as exc:
                        raise ExecutionError(
                            "spec.data.image_embedding resolved invalid "
                            "image_group_sizes from upstream node "
                            f"{source_node!r}."
                        ) from exc

        return InferenceEntry(
            task_id=task_id,
            prompts=prompts,
            inference_cfg=inference_cfg,
            data_cfg=copy.deepcopy(data),
            metadata_raw=metadata_raw,
            append_system_prompt=append_system_prompt,
            images=images,
            image_group_sizes=image_group_sizes,
            image_embedding_path=image_embedding,
            tables=table_stores_list,
        )

    def _populate_table(
        self,
        payload: dict[str, Any],
        table_stores_list: list[pd.DataFrame],
    ):
        """
        Group row-level generation outputs back into per-table outputs.
        """
        items = payload["items"]
        cur = 0
        grouped_items: list[dict[str, Any]] = []
        for df in table_stores_list:
            if not isinstance(df, pd.DataFrame):
                raise ExecutionError("table_stores_list must contain DataFrames.")
            size = len(df)
            outputs = [item["output"] for item in items[cur : cur + size]]
            grouped_items.append({"output": outputs})
            cur += size
        if cur != len(items):
            raise ExecutionError(
                f"Output length {len(items)} does not match "
                f"the total number of rows {cur} in table stores."
            )
        payload["items"] = grouped_items
        return payload

    def _maybe_apply_dataset_shard(self, dataset, spec: TaskSpecStrictBase):
        shard_cfg = spec.shard
        if shard_cfg is None:
            return dataset
        total = shard_cfg.total
        index = shard_cfg.index
        if total <= 1:
            return dataset
        contiguous = True if shard_cfg.contiguous is None else shard_cfg.contiguous
        try:
            return dataset.shard(num_shards=total, index=index, contiguous=contiguous)
        except Exception as exc:
            raise ExecutionError(f"Failed to shard dataset ({index}/{total}): {exc}")
