"""JSONL read / write helpers shared by server, worker, and SDK."""

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file and yield decoded dict rows. Missing file → empty."""
    if not path.exists() or not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        yield from parse_jsonl_lines(fh)


def encode_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> Iterator[bytes]:
    """Encode dict rows as JSONL bytes: one row per line, trailing newline."""
    for row in rows:
        yield (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
