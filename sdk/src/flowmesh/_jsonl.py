"""JSONL parsing helpers for streamed trace responses (sync + async)."""

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any


def parse_jsonl_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield decoded dict rows from text lines; skip empty / malformed."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


async def aparse_jsonl_lines(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Async variant of :func:`parse_jsonl_lines` for streamed responses."""
    async for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
