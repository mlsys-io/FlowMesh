"""Helpers for downloading stage artifacts required by training executors."""

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from shared.utils.http import add_auth_headers

from ..base_executor import ExecutionError


def resolve_jsonl_path(
    path_value: str,
    *,
    out_dir: Path,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    logger=None,
) -> Path:
    """Ensure a JSONL reference is available locally and return the path.

    Supports local filesystem paths and HTTP(S) URLs. When a URL is provided the
    content is downloaded into ``out_dir / "inputs"`` before returning the local
    filename.
    """

    if not path_value:
        raise ExecutionError("JSONL path is empty")

    value = str(path_value).strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        target_dir = (out_dir / "inputs").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or "dataset.jsonl"
        target_path = (target_dir / filename).resolve()
        request_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        add_auth_headers(request_headers)
        try:
            with requests.get(
                value, headers=request_headers, timeout=timeout, stream=True
            ) as response:
                response.raise_for_status()
                with target_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:
            raise ExecutionError(
                f"Failed to download JSONL dataset from {value}: {exc}"
            ) from exc

        if logger:
            try:
                size = target_path.stat().st_size
            except OSError:
                size = None
            logger.info(
                "Downloaded JSONL dataset from %s to %s%s",
                value,
                target_path,
                f" ({size} bytes)" if size is not None else "",
            )
        return target_path

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        rel_candidate = (out_dir / candidate).resolve()
        if rel_candidate.exists():
            candidate = rel_candidate
        else:
            candidate = candidate.resolve(strict=False)

    if candidate.exists():
        return candidate

    search_paths = [candidate]
    base_dir = Path(out_dir).resolve()
    for parent in itertools.islice(base_dir.parents, 0, 2):
        guess = (parent / candidate.name).resolve()
        search_paths.append(guess)
        if guess.exists():
            return guess

    raise ExecutionError(f"JSONL dataset not found: {candidate}")


def normalize_prompt_payload(
    items: list[Any],
) -> tuple[list[str | Sequence[dict[str, str]]], bool, bool]:
    """Validate and normalize a prompt payload.

    Prompt payloads can be either a list of strings or a list of chat-style messages
    (dicts with 'role' and 'content' string fields). Returns the normalized prompts
    along with booleans indicating whether to apply chat template and whether a system
    prompt was found.
    """
    if not items:
        return [], False, False

    first_item = items[0]
    item_type = type(first_item)
    if not (item_type in (str, list) and all(isinstance(x, item_type) for x in items)):
        raise ExecutionError(
            "spec.data.items must be a homogeneous list of strings or lists of dicts "
            "for type == 'list' when fetch_images is False."
        )

    if item_type is str:
        return items, False, False

    found_system_prompt = None
    normalized: list[str | Sequence[dict[str, str]]] = []
    for messages in items:
        if not isinstance(messages, list) or not messages:
            raise ExecutionError(
                "Each entry in spec.data.items must be a non-empty list of "
                "messages (dicts with 'role' and 'content' string fields) when "
                "type == 'list' and fetch_images is False."
            )
        normalized_messages: list[dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                raise ExecutionError(
                    "Each message must be a dict with 'role' and 'content' string "
                    "fields."
                )
            role = m.get("role")
            content = m.get("content")
            if not (isinstance(role, str) and isinstance(content, str)):
                raise ExecutionError(
                    "Each message must have string 'role' and 'content' fields."
                )
            normalized_messages.append({"role": role, "content": content})
        if normalized_messages[-1]["role"] != "user":
            raise ExecutionError(
                "The last message in each item must have the role 'user'."
            )
        has_system = normalized_messages[0]["role"] == "system"
        if found_system_prompt is None:
            found_system_prompt = has_system
        elif found_system_prompt:
            if not has_system:
                raise ExecutionError(
                    "When a system prompt is present, the first message in each item "
                    "must have the role 'system'."
                )
        elif has_system:
            raise ExecutionError(
                "When a system prompt is not present, the first message in each item "
                "must not have the role 'system'."
            )
        normalized.append(normalized_messages)

    return normalized, True, bool(found_system_prompt)
