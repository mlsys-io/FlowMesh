"""Tests for _evaluate_expr attribute/index resolution over pydantic models,
aliases, and nested lists."""

from shared.schemas.result import APIGroupItem, APIItem, APIResult
from worker.executors.utils.graph_templates import _evaluate_expr


def _item(content: str) -> APIItem:
    item = APIItem(index=0, url="u", status_code=200)
    item.response_json = {"choices": [{"message": {"content": content}}]}
    return item


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
    value = _evaluate_expr("Up.items.json.choices[0].message.content", {"Up": upstream})
    assert value == ["c0", "c1"]


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
    value = _evaluate_expr(
        "Up.items.rows.json.choices[0].message.content", {"Up": upstream}
    )
    assert value == [["c0", "c1"], ["c2"]]
