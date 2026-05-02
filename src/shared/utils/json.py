import datetime
import json
import uuid
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


def to_json_serializable(val: Any) -> Any:
    """Simplify value for JSON serialization, specifically handling dates."""
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    return val


def safe_get(data: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def validate_keys(
    obj: dict,
    path: str,
    *,
    allowed: set[str] | None = None,
    required: set[str] | None = None,
) -> None:
    assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
    actual_keys = set(obj.keys())
    if allowed:
        unexpected = actual_keys - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected keys at {path}: {sorted(unexpected)}. "
                f"Allowed: {sorted(allowed)}"
            )
    if required:
        missing = required - actual_keys
        if missing:
            raise KeyError(f"Missing required keys at {path}: {sorted(missing)}")


def dedup_json(obj: Any) -> dict[str, dict[str, Any]]:
    """
    Walk `obj` and replace every string value with a reference:
        {"__dedup_ref__": "<uuid>"}
    Maintain a mapping uuid -> original string in `uuid_to_content`.
    """
    content_to_uuid: dict[str, str] = {}
    uuid_to_content: dict[str, str] = {}

    def _dedup(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _dedup(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_dedup(x) for x in node]
        elif isinstance(node, str):
            # Reuse UUID if we've already seen this exact string
            if node not in content_to_uuid:
                uid = str(uuid.uuid4())
                content_to_uuid[node] = uid
                uuid_to_content[uid] = node
            else:
                uid = content_to_uuid[node]
            return {"__dedup_ref__": uid}
        else:
            # Numbers, bools, null, etc. pass through unchanged
            return node

    deduped_obj = _dedup(obj)
    return {
        "content": uuid_to_content,  # uuid -> original string
        "data": deduped_obj,  # original structure with refs
    }


def _restore_deduped_node(node: Any, uuid_to_content: dict[str, str]) -> Any:
    if isinstance(node, dict):
        # Detect a reference wrapper
        if node.keys() == {"__dedup_ref__"}:
            uid = node["__dedup_ref__"]
            try:
                return uuid_to_content[uid]
            except KeyError:
                raise KeyError(f"Unknown dedup reference UUID: {uid!r}")
        else:
            return {
                k: _restore_deduped_node(v, uuid_to_content) for k, v in node.items()
            }
    elif isinstance(node, list):
        return [_restore_deduped_node(x, uuid_to_content) for x in node]
    else:
        return node


def restore_json(deduped_json: dict[str, dict[str, Any]]) -> Any:
    """
    Inverse of dedup_json: replace
        {"__dedup_ref__": "<uuid>"}
    with the original string from uuid_to_content.
    """
    return _restore_deduped_node(deduped_json["data"], deduped_json["content"])


def lookup_deduped_json(deduped_json: dict[str, dict[str, Any]], key: str) -> str:
    """Given a deduped JSON structure, look up the original string for a given key."""
    return _restore_deduped_node(deduped_json["data"][key], deduped_json["content"])


def normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
