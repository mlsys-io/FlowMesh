"""Utilities for building inference prompts that combine upstream graph outputs."""

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel

from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.utils.json import validate_keys

from ...utils.serialization import try_deserialize_dataframe
from ..base_executor import ExecutionError
from .safe_eval import safe_execute_function, safe_materialize_function

_MISSING: Any = object()

type MessageItem = dict[str, str]
type Message = Sequence[MessageItem]
type MaterializedMessage = Sequence[MessageItem]
type MaterializedMessageOrTable = (
    MaterializedMessage | pd.DataFrame | dict[str, dict[str, Any]]
)


def build_prompts_from_graph_template(
    data_cfg: dict[str, Any], spec: TaskSpecStrictBase
) -> list[str | Message]:
    """Render prompt strings from a `graph_template` data configuration.

    Args:
        data_cfg: The `spec.data` section for the task.
        spec:     The full `spec` object (expected to contain `_upstreamResults`).

    Returns:
        A list of prompts ready to be consumed by the inference executor.
    """

    if not isinstance(data_cfg, dict):
        raise ExecutionError(
            "spec.data must be a mapping when type == 'graph_template'."
        )

    validate_keys(
        data_cfg, "spec.data", allowed={"type", "template", "image_embedding"}
    )
    template_cfg = data_cfg.get("template") or {}
    if not isinstance(template_cfg, dict):
        raise ExecutionError(
            "spec.data.template must be a mapping for graph templates."
        )
    validate_keys(
        template_cfg,
        "spec.data.template",
        allowed={"name", "text", "columns", "options"},
    )

    context = spec.upstreamResults or {}

    columns_cfg = template_cfg.get("columns") or []
    if not context and any(col.get("expr") for col in columns_cfg):
        raise ExecutionError(
            "No upstream results available for graph template prompts, yet "
            "expressions were specified."
        )

    columns = _resolve_columns(columns_cfg, context)

    name = str(template_cfg.get("name") or "two_column_briefing")
    options = template_cfg.get("options") or {}

    renderer = _TEMPLATE_REGISTRY.get(name)
    if renderer is None:
        text = template_cfg.get("text")
        if isinstance(text, str) and text.strip():
            return [_render_inline_text(text, columns, options)]
        raise ExecutionError(f"Unknown graph template '{name}'.")

    prompt = renderer(columns, options)
    return _maybe_broadcast_image_prompts(prompt, data_cfg, context)


def _maybe_broadcast_image_prompts(
    prompts: Sequence[str | Message],
    data_cfg: dict[str, Any],
    context: dict[str, Any],
) -> list[str | Message]:
    image_embedding = data_cfg.get("image_embedding")
    if not isinstance(image_embedding, dict):
        return list(prompts)
    if not prompts or len(prompts) != 1:
        return list(prompts)
    count = _resolve_image_embedding_count(image_embedding, context)
    if not count or count <= 1:
        return list(prompts)
    return list(prompts) * count


def _resolve_image_embedding_count(
    image_embedding: dict[str, Any], context: dict[str, Any]
) -> int | None:
    node = image_embedding.get("node")
    if not isinstance(node, str):
        expr = image_embedding.get("expr")
        if isinstance(expr, str) and expr.strip():
            node = expr.split(".", 1)[0].strip()
    if not isinstance(node, str) or not node:
        return None
    upstream = context.get(node)
    if not isinstance(upstream, dict):
        return None
    count = upstream.get("count")
    if isinstance(count, int) and count > 0:
        return count
    return None


def _resolve_columns(columns_cfg: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(columns_cfg, list):
        raise ExecutionError("graph_template.template.columns must be a list.")

    columns: list[Any] = []
    for idx, raw in enumerate(columns_cfg):
        if not isinstance(raw, dict):
            raise ExecutionError("Each column definition must be a mapping.")

        label = str(raw.get("label") or f"Column {idx + 1}").strip()

        expr = raw.get("expr")
        if not expr:
            node = raw.get("node")
            path = raw.get("path")
            if node and path:
                expr = f"{node}.{path}"

        data = raw.get("data")

        if expr:
            assert data is None
            value = _evaluate_expr(expr.strip(), context)
            if value is None:
                if "default" in raw:
                    value = raw.get("default")
                else:
                    raise ExecutionError(
                        f"Column '{label}' expression '{expr}' resolved to null."
                    )
        elif data:
            assert expr is None
            dtype: str = data["type"]
            match dtype:
                case "dataset":  # TODO: Support dataset type
                    raise ExecutionError(
                        "Column 'data' type == 'dataset' is not supported yet."
                    )
                case "list":
                    items = data.get("items")
                    if not isinstance(items, list):
                        raise ExecutionError("data.items must be a list.")
                    value = items
                case "dataframe":
                    nested_columns_cfg = data.get("columns")
                    nested_columns = _resolve_columns(nested_columns_cfg, context)
                    value = _build_grouped_dataframes(nested_columns)
                case _:
                    raise ExecutionError(f"Unsupported column 'data' type: {dtype}")
        else:
            raise ExecutionError(
                f"Column '{label}' is missing an expr/node+path/data definition."
            )

        columns.append(
            {
                "label": label,
                "value": value,
                "expr": expr,
            }
        )

    return columns


def _build_grouped_dataframes(columns: list[dict[str, Any]]) -> list[pd.DataFrame]:
    if not columns:
        raise ExecutionError("dataframe data type requires at least one column.")

    grouped_columns: dict[str, list[list[Any]]] = {}
    for column in columns:
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
                "dataframe column values must resolve to the same number of groups."
            )

    dataframes: list[pd.DataFrame] = []
    for group_idx in range(group_count):
        max_len = 1
        raw_values: dict[str, list[Any]] = {}
        for label, groups in grouped_columns.items():
            values = groups[group_idx]
            if not isinstance(values, list):
                values = [values]
            if values:
                max_len = max(max_len, len(values))
            raw_values[label] = values

        normalized: dict[str, list[Any]] = {}
        for label, values in raw_values.items():
            if len(values) == 1 and max_len > 1:
                values = [values[0] for _ in range(max_len)]
            elif len(values) != max_len:
                raise ExecutionError(
                    "dataframe column values must resolve to "
                    "the same number of rows per group."
                )
            normalized[label] = values
        dataframes.append(pd.DataFrame(normalized))

    return dataframes


def _coerce_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _render_inline_text(
    text: str, columns: list[dict[str, str]], _: dict[str, Any]
) -> str:
    mapping: dict[str, str] = {}
    for idx, column in enumerate(columns):
        mapping[f"col{idx}_label"] = column["label"]
        mapping[f"col{idx}_value"] = _coerce_to_string(column["value"])

    class _SafeDict(dict):
        def __missing__(self, key):  # type: ignore[override]
            return "{" + key + "}"

    return text.format_map(_SafeDict(mapping))


def _aggregate_structural_messages(
    columns: dict[str, Sequence[str | MaterializedMessageOrTable]],
    msg_options: Sequence[dict[str, str]],
) -> Sequence[MaterializedMessage]:
    def _is_expandable_group_value(value: Any) -> bool:
        return isinstance(value, list) and not all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in value
        )

    num_groups = max(len(v) for v in columns.values()) if len(columns) > 0 else 1
    assert all(len(v) == 1 or len(v) == num_groups for v in columns.values())
    grouped_columns = {
        key: (
            values
            if len(values) == num_groups
            else [values[0] for _ in range(num_groups)]
        )
        for key, values in columns.items()
    }

    group_row_counts: list[int] = []
    for group_idx in range(num_groups):
        row_count = 1
        for values in grouped_columns.values():
            group_value = values[group_idx]
            if _is_expandable_group_value(group_value):
                row_count = max(row_count, len(group_value))
        group_row_counts.append(row_count)

    columns = {key: [] for key in grouped_columns}
    for group_idx, row_count in enumerate(group_row_counts):
        for key, values in grouped_columns.items():
            group_value = values[group_idx]
            if _is_expandable_group_value(group_value):
                value_list = list(group_value)
                if len(value_list) == 1 and row_count > 1:
                    value_list = [value_list[0] for _ in range(row_count)]
                elif len(value_list) != row_count:
                    raise ExecutionError(
                        "Grouped graph-template values must resolve to the same "
                        "number of rows per group."
                    )
            else:
                value_list = [group_value for _ in range(row_count)]
            columns[key].extend(value_list)  # type: ignore

    num_rows = sum(group_row_counts)

    batch_messages: list[Message] = [[] for _ in range(num_rows)]

    class _SafeDict(dict):
        def __missing__(self, key):  # type: ignore[override]
            return "{" + key + "}"

    for message_metadata in msg_options:
        if "content" not in message_metadata:
            raise RuntimeError(
                f"Each message must have 'content' field. {message_metadata}"
            )
        raw_content: str = message_metadata["content"]
        if raw_content in columns:
            content = columns[raw_content]  # Materialize Message
        else:
            rendered_rows: list[str] = []
            # Disable pandas width caps so wide DataFrame cells render in full.
            with pd.option_context(
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.max_colwidth",
                None,
            ):
                for row_idx in range(num_rows):
                    row_mapping: dict[str, str] = {}
                    for label, values in columns.items():
                        row_value = values[row_idx]
                        if isinstance(row_value, pd.DataFrame):
                            row_mapping[label] = row_value.to_markdown(index=False)
                        else:
                            row_mapping[label] = _coerce_to_string(row_value)
                    rendered_rows.append(raw_content.format_map(_SafeDict(row_mapping)))
            content = rendered_rows
        if role := message_metadata.get("role"):
            assert all(isinstance(prompt, str) for prompt in content), (
                content,
                columns,
            )
            for messages, prompt in zip(batch_messages, content):
                messages.append({"role": role, "content": prompt})  # type: ignore
        else:
            assert all(isinstance(msg, dict) for prompt in content for msg in prompt), (
                content,
                columns,
            )
            for messages, prompt in zip(batch_messages, content):
                messages.extend(prompt)  # type: ignore
    return batch_messages


def _render_template(
    columns: dict[str, Sequence[str | MaterializedMessageOrTable]],
    template: str,
    format_kwargs: dict[str, str],
) -> list[str | MaterializedMessage]:
    list_lengths = [len(v) for v in columns.values() if len(v) > 1]
    if list_lengths:
        num_rows = list_lengths[0]
        assert all(
            length == list_lengths[0] for length in list_lengths
        ), "All list-type format arguments must have the same length."
    else:
        num_rows = 1

    format_kwargs_materialized: dict[str, list[str | Sequence[MessageItem]]] = {}
    for k, col_id in format_kwargs.items():
        if col_id not in columns:
            raise ExecutionError(
                f"Column '{col_id}' not found for template formatting."
            )

        value_list = (
            columns[col_id]
            if len(columns[col_id]) == num_rows
            else [columns[col_id][0] for _ in range(num_rows)]
        )
        # Disable pandas width caps so wide DataFrame cells render in full.
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            None,
            "display.max_colwidth",
            None,
        ):
            value_list_rendered = [
                v.to_markdown(index=False) if isinstance(v, pd.DataFrame) else v
                for v in value_list
            ]
        format_kwargs_materialized[k] = value_list_rendered  # type: ignore

    prompts: list[str | Sequence[dict[str, str]]] = [
        template.format(
            **{
                k: format_kwargs_materialized[k][i]
                for k in format_kwargs_materialized.keys()
            }
        )
        for i in range(num_rows)
    ]
    return prompts


def _render_lambda_func(
    columns: dict[str, Sequence[str | MaterializedMessageOrTable]],
    fn: str,
    fn_args: Sequence[Message | str],
) -> list[str | MaterializedMessage]:
    list_lengths = [len(v) for v in columns.values() if len(v) > 1]
    if list_lengths:
        num_rows = list_lengths[0]
        assert all(
            length == list_lengths[0] for length in list_lengths
        ), "All list-type function arguments must have the same length."
    else:
        num_rows = 1

    fn_obj = safe_materialize_function(fn)

    materialized_args: list[Sequence[MaterializedMessage | str]] = []
    for arg in fn_args:
        if not isinstance(arg, str):
            materialized_args.append(_aggregate_structural_messages(columns, arg))
            continue

        if arg in columns:
            values = columns[arg]
            if len(values) == 1 and num_rows > 1:
                values = [values[0] for _ in range(num_rows)]
            values_filtered: list[str | MaterializedMessage] = []
            for v in values:
                if isinstance(v, dict):
                    raise ExecutionError(
                        f"Column '{arg}' must not be tables as function argument."
                    )
                values_filtered.append(v)  # type: ignore
            materialized_args.append(values_filtered)
        else:
            materialized_args.append([arg for _ in range(num_rows)])

    prompts: list[str | MaterializedMessage] = []
    for i in range(num_rows):
        args = tuple(values[i] for values in materialized_args)
        try:
            prompt = safe_execute_function(fn_obj, args)
        except Exception as e:
            raise ExecutionError(
                f"Error evaluating graph template lambda function: {e}"
            ) from e
        prompts.append(prompt)
    return prompts


def _render_structural_messages(
    columns: list[dict[str, Any]], options: dict[str, Any]
) -> Sequence[Sequence[dict[str, str]]]:
    """
    This is a powerful graph template renderer that supports multi-step formatting and
    function evaluation to build structured message prompts.

    options should have the following structure:
    format:
        steps (optional):
            - label:
              template|function:
              arguments:
                - label:
                    value:
                - xxxID
        messages:
            - role (optional):
              content:
    """
    format_options = options["format"]
    if any("label" not in column for column in columns):
        raise RuntimeError(
            "Detected positional arguments when constructing structural messages. This "
            "is not supported as we cannot guarantee finding correct item by index. "
        )
    formatted_prompts: dict[str, Sequence[str | MaterializedMessageOrTable]] = {
        column["label"]: (
            column["value"] if isinstance(column["value"], list) else [column["value"]]
        )
        for column in columns
    }
    for step_option in format_options.get("steps", []):
        if "template" in step_option:
            format_kwargs = {
                kwarg["label"]: kwarg["value"]
                for kwarg in step_option.get("arguments", [])
            }
            formatted_prompts[step_option["label"]] = _render_template(
                formatted_prompts, step_option["template"], format_kwargs
            )
        elif "function" in step_option:
            fn_args = step_option.get("arguments", [])
            formatted_prompts[step_option["label"]] = _render_lambda_func(
                formatted_prompts, step_option["function"], fn_args
            )
        else:
            raise RuntimeError(
                "Each step must specify either 'template' or 'function'."
            )

    batch_messages = _aggregate_structural_messages(
        formatted_prompts, format_options["messages"]
    )

    return batch_messages


def _render_two_column_briefing(
    columns: list[dict[str, str]], options: dict[str, Any]
) -> list[str]:
    if len(columns) < 2:
        raise ExecutionError(
            "two_column_briefing template expects at least two columns."
        )

    role = str(options.get("role") or "energy strategist")
    intro_lines = options.get("intro")
    closing_lines = options.get("closing")

    lines: list[str] = []

    if intro_lines:
        if isinstance(intro_lines, list):
            lines.extend(str(x) for x in intro_lines)
        else:
            lines.append(str(intro_lines))
    else:
        article = "an" if role[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        lines.append(
            f"You are {article} {role}. "
            "Combine the following factor analyses side by side"
        )
        lines.append("and produce a concise, comparative briefing:")

    for column in columns:
        lines.append(
            _format_column_line(column["label"], _coerce_to_string(column["value"]))
        )

    if closing_lines:
        if isinstance(closing_lines, list):
            lines.extend(str(x) for x in closing_lines)
        else:
            lines.append(str(closing_lines))
    else:
        left_label = columns[0]["label"].lower()
        right_label = columns[1]["label"].lower()
        lines.append(
            "Present them as a two-column style summary with actionable "
            "recommendations that"
        )
        lines.append(f"weigh the {left_label} against the {right_label}.")

    return ["\n".join(lines)]


def _format_column_line(label: str, value: str) -> str:
    cleaned = value.replace("\r\n", "\n").strip()
    if "\n" in cleaned:
        indented = cleaned.replace("\n", "\n  ")
    else:
        indented = cleaned
    return f"• {label}: {indented}"


def _evaluate_expr(expr: str, context: dict[str, BaseExecutorResult]) -> Any:
    if not expr:
        return None

    parts = expr.split(".")
    root = parts[0]
    result = context.get(root)
    if result is None:
        return None

    value: Any = result
    for token in parts[1:]:
        if not token:
            continue
        attr, indexes = _split_indexes(token)
        if attr:
            if isinstance(value, dict) and attr in value:
                value = value[attr]
            elif isinstance(value, list) and all(
                isinstance(v, dict) and attr in v for v in value
            ):
                value = [v[attr] for v in value]
            elif isinstance(value, list) and all(
                isinstance(v, pd.DataFrame) for v in value
            ):
                if any(attr not in v.columns for v in value):
                    raise ExecutionError(
                        f"{attr} not a valid column in one of the "
                        f"DataFrames for {token}."
                    )
                value = [v[attr].tolist() for v in value]
            elif isinstance(value, pd.DataFrame):
                if attr not in value.columns:
                    raise ExecutionError(
                        f"{attr} not a valid column in DataFrame for {token}."
                    )
                value = value[attr].tolist()
            elif isinstance(value, BaseModel):
                resolved = getattr(value, attr, _MISSING)
                if resolved is _MISSING:
                    raise ExecutionError(
                        f"{attr} not a valid attribute of {type(value).__name__} "
                        f"for {token}."
                    )
                value = resolved
            else:
                raise ExecutionError(
                    f"{attr} in {parts} is not a valid key - "
                    f"{type(value).__name__}, {value}"
                )
        for idx in indexes:
            if isinstance(value, list) and -len(value) <= idx < len(value):
                value = value[idx]
            elif isinstance(value, list) and all(isinstance(v, list) for v in value):
                value = [v[idx] for v in value]
            else:
                raise ExecutionError(
                    f"{idx} not a valid index in {token} - {len(value)}"
                )
        # Attempt to deserialize DataFrame if applicable
        if isinstance(value, dict):
            value = try_deserialize_dataframe(value)
        elif isinstance(value, list) and all(isinstance(v, dict) for v in value):
            value = [try_deserialize_dataframe(v) for v in value]
    return value


def _split_indexes(token: str) -> tuple[str, list[int]]:
    parts = token.split("[")
    attr = parts[0]
    idx_list: list[int] = []
    for part in parts[1:]:
        part = part.rstrip("]")
        if part:
            try:
                idx_list.append(int(part))
            except ValueError:
                idx_list.append(-1)
    return attr, idx_list


_TEMPLATE_REGISTRY: dict[str, Any] = {
    "two_column_briefing": _render_two_column_briefing,
    "format": _render_structural_messages,
}
