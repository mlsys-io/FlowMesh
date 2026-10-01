"""Tests for _evaluate_expr attribute/index resolution over pydantic models,
aliases, and nested lists."""

from typing import cast

import pandas as pd

from shared.schemas.result import (
    APIGroupItem,
    APIItem,
    APIResult,
    BaseExecutorResult,
    DataRetrievalItem,
    DataRetrievalResult,
    InferenceItem,
    InferenceResult,
)
from worker.executors.utils.graph_templates import (
    _aggregate_structural_messages,
    _build_grouped_dataframes,
    _evaluate_expr,
)


def _item(content: str) -> APIItem:
    item = APIItem(index=0, url="u", status_code=200)
    item.response_json = {"choices": [{"message": {"content": content}}]}
    return item


def _messages(
    columns: dict, grouped_labels: set[str], content: str = "row {L}"
) -> list[str]:
    """Render one user message per row over the given columns."""
    batch = _aggregate_structural_messages(
        columns,
        [{"role": "user", "content": content}],
        grouped_labels,
    )
    return [m["content"] for message in batch for m in message]


def test_aggregate_ungrouped_list_column_expands_per_row() -> None:
    """An ungrouped list column over 2 groups of 2 and 3 rows yields 5 messages
    in order, one per row."""
    columns = {"L": [["c0", "c1"], ["c2", "c3", "c4"]]}
    assert _messages(columns, set()) == [
        "row c0",
        "row c1",
        "row c2",
        "row c3",
        "row c4",
    ]


def test_aggregate_grouped_column_keeps_whole_group() -> None:
    """A grouped column over the same data yields 2 messages, each carrying its
    whole group."""
    columns = {"L": [["c0", "c1"], ["c2", "c3", "c4"]]}
    assert _messages(columns, {"L"}) == [
        'row ["c0", "c1"]',
        'row ["c2", "c3", "c4"]',
    ]


def test_aggregate_mixed_grouped_and_ungrouped_columns() -> None:
    """A grouped column mixed with an ungrouped list column yields one message
    per ungrouped row, each carrying the whole grouped value."""
    columns = {
        "G": [["g0", "g1"], ["g2", "g3", "g4"]],
        "L": [["c0", "c1"], ["c2", "c3", "c4"]],
    }
    assert _messages(columns, {"G"}, "row {G} {L}") == [
        'row ["g0", "g1"] c0',
        'row ["g0", "g1"] c1',
        'row ["g2", "g3", "g4"] c2',
        'row ["g2", "g3", "g4"] c3',
        'row ["g2", "g3", "g4"] c4',
    ]


def test_expr_over_list_of_models_with_alias() -> None:
    """Attribute access maps over a list of APIItem models, resolving the
    ``json`` alias to ``response_json``."""
    upstream = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://up.example.com",
        status_code=200,
        items=[_item("c0"), _item("c1")],
    )
    value, grouped = _evaluate_expr(
        "Up.items.json.choices[0].message.content", {"Up": upstream}
    )
    assert value == ["c0", "c1"]
    assert grouped is False


def test_expr_over_nested_lists_of_models() -> None:
    """Attribute access maps through nested lists (a list of groups, each a
    list of APIItem models), yielding one inner list per group."""
    upstream = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://up.example.com",
        status_code=200,
        items=[
            APIGroupItem(index=0, rows=[_item("c0"), _item("c1")]),
            APIGroupItem(index=1, rows=[_item("c2")]),
        ],
    )
    value, grouped = _evaluate_expr(
        "Up.items.rows.json.choices[0].message.content", {"Up": upstream}
    )
    assert value == [["c0", "c1"], ["c2"]]
    assert grouped is True


def test_build_grouped_dataframes_all_empty_columns_yield_zero_rows() -> None:
    """All-empty grouped columns yield a zero-row DataFrame, not a mismatch."""
    columns = [
        {"label": "text", "value": []},
        {"label": "statement", "value": []},
    ]
    dataframes = _build_grouped_dataframes(columns)
    assert len(dataframes) == 1
    assert dataframes[0].empty
    assert list(dataframes[0].columns) == ["text", "statement"]
    assert isinstance(dataframes[0], pd.DataFrame)


def _grouped_value(expr: str, upstream: object) -> tuple[list[list[object]], bool]:
    value, grouped = _evaluate_expr(
        expr,
        cast("dict[str, BaseExecutorResult]", {"Up": upstream}),
    )
    assert grouped is True
    assert isinstance(value, list) and all(isinstance(v, list) for v in value)
    return value, grouped


def test_api_rows_groups_by_item() -> None:
    """An upstream API task's ``items.rows`` is grouped: one list per item."""
    upstream = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://up.example.com",
        status_code=200,
        items=[
            APIGroupItem(index=0, rows=[_item("c0"), _item("c1")]),
            APIGroupItem(index=1, rows=[_item("c2")]),
        ],
    )
    value, _ = _grouped_value("Up.items.rows.json.choices[0].message.content", upstream)
    assert value == [["c0", "c1"], ["c2"]]


def test_python_output_groups_by_item() -> None:
    """A python stage whose items each carry a list of records in ``output`` is
    grouped: ``items.output.<field>`` yields one list per item."""
    upstream = {
        "items": [
            {"output": [{"q": "a"}, {"q": "b"}]},
            {"output": [{"q": "c"}]},
        ]
    }
    value, _ = _grouped_value("Up.items.output.q", upstream)
    assert value == [["a", "b"], ["c"]]


def test_vllm_output_groups_by_item() -> None:
    """A vLLM ``_populate_table`` result's ``items.output`` is grouped: one list
    of string outputs per item."""
    upstream = InferenceResult(
        ok=True,
        items=[
            InferenceItem(
                index=0, prompt="p0", output=["s0", "s1"], finish_reason=None
            ),
            InferenceItem(index=1, prompt="p1", output=["s2"], finish_reason=None),
        ],
    )
    value, _ = _grouped_value("Up.items.output", upstream)
    assert value == [["s0", "s1"], ["s2"]]


def test_s3_content_groups_by_item() -> None:
    """An S3 data-retrieval result's ``items.content`` is grouped: one list per
    item."""
    upstream = DataRetrievalResult(
        ok=True,
        items=[
            DataRetrievalItem(index=0, content=["h0", "h1"]),
            DataRetrievalItem(index=1, content=["h2"]),
        ],
    )
    value, _ = _grouped_value("Up.items.content", upstream)
    assert value == [["h0", "h1"], ["h2"]]


def test_per_row_scalar_list_stays_one_cell() -> None:
    """A per-row list of scalars (e.g. tags) stays one cell: the grouping access
    is ``items.output``, and a further attribute over the groups never
    re-evaluates grouping from the inner shape."""
    upstream = {
        "items": [
            {"output": [{"q": "a", "tags": ["x", "y"]}, {"q": "b", "tags": []}]},
            {"output": [{"q": "c", "tags": ["z"]}]},
        ]
    }
    value, _ = _grouped_value("Up.items.output.q", upstream)
    assert value == [["a", "b"], ["c"]]
    tags, grouped = _evaluate_expr(
        "Up.items.output.tags",
        cast("dict[str, BaseExecutorResult]", {"Up": upstream}),
    )
    assert grouped is True
    assert tags == [[["x", "y"], []], [["z"]]]


def test_list_of_dataframes_groups_by_item() -> None:
    """Mapping an attribute over a list of DataFrames (one per item) is grouped,
    so ``items.table.<col>`` builds one table per item rather than one row with
    list cells."""
    upstream = {
        "items": [
            {"table": {"df": pd.DataFrame({"title": ["t0", "t1"]}).to_json()}},
            {"table": {"df": pd.DataFrame({"title": ["t2"]}).to_json()}},
        ]
    }
    value, grouped = _evaluate_expr(
        "Up.items.table.title",
        cast("dict[str, BaseExecutorResult]", {"Up": upstream}),
    )
    assert grouped is True
    assert value == [["t0", "t1"], ["t2"]]
