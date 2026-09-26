"""DataMixin dataframe grouping: a per-row list column is a cell value, not a
group. Grouping is decided from the upstream structure (APIGroupItem.rows),
never from the shape of the cell values (FM-7)."""

from types import SimpleNamespace
from typing import Any, cast

from shared.schemas.result import APIGroupItem, APIItem, APIResult, BaseExecutorResult
from worker.executors.mixins.data import DataMixin


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""


def _row(output: dict[str, Any]) -> APIItem:
    item = APIItem(index=0, url="u", status_code=200)
    item.response_json = {"output": output}
    return item


def _plain_upstream(rows: list[dict[str, Any]]) -> BaseExecutorResult:
    """A non-grouped upstream result: one row dict per item, each carrying an
    ``output`` mapping (the live GatherPremises shape)."""
    return BaseExecutorResult.model_validate(
        {
            "items": [{"output": r} for r in rows],
            "count": len(rows),
        }
    )


def _grouped_upstream(groups: list[list[dict[str, Any]]]) -> APIResult:
    """A grouped upstream api result: one APIGroupItem per group."""
    return APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://up.example.com",
        status_code=200,
        items=[
            APIGroupItem(index=i, rows=[_row(r) for r in group])
            for i, group in enumerate(groups)
        ],
    )


def _spec(
    columns: list[dict[str, Any]],
    upstream: Any,
    content: str = "row {claim} {premises}",
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            data={
                "type": "dataframe",
                "columns": columns,
                "messages": [{"role": "user", "content": content}],
            },
            inference={},
            upstreamResults={"Up": upstream},
        ),
    )


def _collect(
    columns: list[dict[str, Any]],
    upstream: Any,
    content: str = "row {claim} {premises}",
):
    return _Mixin()._collect_prompts_for_spec(
        _spec(columns, upstream, content), "tsk-fm7"
    )


def test_per_row_list_column_is_one_group_with_lists_intact() -> None:
    """A column whose per-row value is a list (mixed lengths, incl. empty) is a
    single group with one row per upstream row, the list intact in each cell."""
    upstream = _plain_upstream(
        [
            {"claim": "c0", "premises": ["p0"]},
            {"claim": "c1", "premises": []},
            {"claim": "c2", "premises": ["p2"]},
        ]
    )
    columns = [
        {"label": "claim", "node": "Up", "path": "items.output.claim"},
        {"label": "premises", "node": "Up", "path": "items.output.premises"},
    ]

    entry = _collect(columns, upstream)

    assert len(entry.tables) == 1
    df = entry.tables[0]
    assert list(df.columns) == ["claim", "premises"]
    assert len(df) == 3
    assert df["claim"].tolist() == ["c0", "c1", "c2"]
    assert df["premises"].tolist() == [["p0"], [], ["p2"]]


def test_per_row_single_member_list_is_not_repeated() -> None:
    """When every per-row list has one member, the column is still one group of
    three rows, not three groups each broadcast to all claims."""
    upstream = _plain_upstream(
        [
            {"claim": "c0", "premises": ["x0"]},
            {"claim": "c1", "premises": ["x1"]},
            {"claim": "c2", "premises": ["x2"]},
        ]
    )
    columns = [
        {"label": "claim", "node": "Up", "path": "items.output.claim"},
        {"label": "premises", "node": "Up", "path": "items.output.premises"},
    ]

    entry = _collect(columns, upstream)

    assert len(entry.tables) == 1
    df = entry.tables[0]
    assert len(df) == 3
    assert df["claim"].tolist() == ["c0", "c1", "c2"]
    assert df["premises"].tolist() == [["x0"], ["x1"], ["x2"]]


def test_genuinely_grouped_upstream_still_groups() -> None:
    """A genuinely grouped upstream result (APIGroupItem.rows) still produces
    one table per group."""
    upstream = _grouped_upstream(
        [
            [{"claim": "c0"}, {"claim": "c1"}],
            [{"claim": "c2"}],
        ]
    )
    columns = [
        {"label": "claim", "node": "Up", "path": "items.rows.json.output.claim"},
    ]

    entry = _collect(columns, upstream, content="row {claim}")

    assert len(entry.tables) == 2
    assert [len(df) for df in entry.tables] == [2, 1]
    assert entry.tables[0]["claim"].tolist() == ["c0", "c1"]
    assert entry.tables[1]["claim"].tolist() == ["c2"]


def _list_mode_lambda_upstream(
    groups: list[list[dict[str, Any]]],
) -> BaseExecutorResult:
    """A list-mode lambda upstream: one item per group, each item's ``output``
    is itself a list of records (the live PairSources shape)."""
    return BaseExecutorResult.model_validate(
        {
            "items": [{"output": group} for group in groups],
            "count": len(groups),
        }
    )


def test_list_mode_lambda_output_groups_by_item() -> None:
    """A list-mode lambda upstream whose items' output is a list of records,
    read as ``items.output.<field>`` columns, gives one group per item with the
    right row counts (uneven group sizes included)."""
    upstream = _list_mode_lambda_upstream(
        [
            [{"claim": "c0", "src_text": "s0"}, {"claim": "c0", "src_text": "s1"}],
            [{"claim": "c1", "src_text": "s2"}],
            [
                {"claim": "c2", "src_text": "s3"},
                {"claim": "c2", "src_text": "s4"},
                {"claim": "c2", "src_text": "s5"},
            ],
        ]
    )
    columns = [
        {"label": "claim", "node": "Up", "path": "items.output.claim"},
        {"label": "src_text", "node": "Up", "path": "items.output.src_text"},
    ]

    entry = _collect(columns, upstream, content="row {claim} {src_text}")

    assert len(entry.tables) == 3
    assert [len(df) for df in entry.tables] == [2, 1, 3]
    assert entry.tables[0]["claim"].tolist() == ["c0", "c0"]
    assert entry.tables[0]["src_text"].tolist() == ["s0", "s1"]
    assert entry.tables[1]["claim"].tolist() == ["c1"]
    assert entry.tables[2]["claim"].tolist() == ["c2", "c2", "c2"]


def test_index_mapping_over_nested_list_stays_one_group() -> None:
    """``items.json.choices[0].message.content`` over a non-grouped api result
    stays one group: the index mapping over a per-item list is not grouping."""
    items: list[APIItem | APIGroupItem] = []
    for i in range(3):
        item = APIItem(index=i, url="u", status_code=200)
        item.response_json = {"choices": [{"message": {"content": f"c{i}"}}]}
        items.append(item)
    upstream = APIResult(
        ok=True,
        executor="api",
        method="POST",
        url="https://up.example.com",
        status_code=200,
        items=items,
    )
    columns = [
        {
            "label": "content",
            "node": "Up",
            "path": "items.json.choices[0].message.content",
        },
    ]

    entry = _collect(columns, upstream, content="row {content}")

    assert len(entry.tables) == 1
    df = entry.tables[0]
    assert len(df) == 3
    assert df["content"].tolist() == ["c0", "c1", "c2"]
