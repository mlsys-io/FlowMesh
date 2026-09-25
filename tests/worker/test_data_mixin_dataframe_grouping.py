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
