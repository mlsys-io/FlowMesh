"""DataMixin._populate_table: grouping rows into per-table InferenceItems."""

from typing import Any

import pandas as pd
import pytest

from shared.schemas.result import InferenceItem
from worker.executors.base_executor import ExecutionError
from worker.executors.mixins.data import DataMixin


class _Mixin(DataMixin):
    """Bare-bones DataMixin instance for unit testing."""


def _row(
    index: int,
    text: str,
    *,
    finish: str | None = "stop",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index,
        "prompt": f"prompt-{index}",
        "output": text,
        "finish_reason": finish,
    }
    if metadata is not None:
        row["metadata"] = metadata
    return row


def _tables(*sizes: int) -> list[pd.DataFrame]:
    return [pd.DataFrame({"col": list(range(n))}) for n in sizes]


def test_grouped_items_validate_as_inference_items() -> None:
    """Grouped payloads are consumed as InferenceItems downstream."""
    items = [_row(i, f"out-{i}") for i in range(3)]

    grouped = _Mixin()._populate_table(items, _tables(2, 1))

    assert len(grouped) == 2
    for payload in grouped:
        InferenceItem.model_validate(payload)


def test_outputs_are_grouped_per_table_in_order() -> None:
    items = [_row(i, f"out-{i}") for i in range(3)]

    grouped = _Mixin()._populate_table(items, _tables(2, 1))

    assert [p["output"] for p in grouped] == [["out-0", "out-1"], ["out-2"]]
    assert [p["index"] for p in grouped] == [0, 1]
    assert [p["prompt"] for p in grouped] == ["prompt-0", "prompt-2"]


def test_finish_reason_is_a_list_covering_every_member() -> None:
    items = [_row(0, "a"), _row(1, "b", finish=None), _row(2, "c", finish="length")]

    grouped = _Mixin()._populate_table(items, _tables(2, 1))

    assert grouped[0]["finish_reason"] == ["stop", None]
    assert grouped[1]["finish_reason"] == ["length"]
    for payload in grouped:
        InferenceItem.model_validate(payload)


def test_metadata_is_carried_not_dropped() -> None:
    items = [_row(0, "a", metadata={"episode_id": "ep-0"}), _row(1, "b")]

    grouped = _Mixin()._populate_table(items, _tables(2))

    assert grouped[0]["metadata"] == {"episode_id": "ep-0"}


def test_absent_metadata_is_omitted_rather_than_null() -> None:
    grouped = _Mixin()._populate_table([_row(0, "a")], _tables(1))

    assert "metadata" not in grouped[0]
    InferenceItem.model_validate(grouped[0])


def test_row_count_mismatch_still_raises() -> None:
    with pytest.raises(ExecutionError):
        _Mixin()._populate_table([_row(0, "a")], _tables(2))


def test_non_dataframe_still_raises() -> None:
    with pytest.raises(ExecutionError):
        _Mixin()._populate_table([_row(0, "a")], ["not-a-dataframe"])  # type: ignore[list-item]
