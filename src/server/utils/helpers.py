import asyncio
import json
import logging
import queue
import threading
import uuid
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from typing import Any

import docker
from redis.client import PubSub

_logger: logging.Logger | None = None
_docker_client: docker.DockerClient | None = None


def get_logger(
    name: str = "server",
    log_file: str = "server.log",
    max_bytes: int = 0,
    backup_count: int = 0,
    level: str = "INFO",
) -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fh = RotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    _logger = logger
    return logger


def get_docker_client() -> docker.DockerClient:
    global _docker_client
    if _docker_client is not None:
        return _docker_client

    _docker_client = docker.from_env()
    return _docker_client


class TSQueue[T]:
    def __init__(self) -> None:
        self._q: queue.Queue[T] = queue.Queue()

    async def put(self, item: T) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._q.put, item)

    async def get(self) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._q.get)


class ResourcePool[T]:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reserved: set[T] = set()

    def reserve(self, item: T) -> bool:
        with self._lock:
            if item in self._reserved:
                return False
            self._reserved.add(item)
            return True

    def release(self, item: T) -> None:
        with self._lock:
            self._reserved.discard(item)


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
    """Given a deduped JSON structure, look up the original content for a given key."""
    return _restore_deduped_node(deduped_json["data"][key], deduped_json["content"])


def normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def iter_pubsub_messages(pubsub: PubSub) -> Iterable[Any]:
    """Iterate over messages from a Redis PubSub instance."""
    try:
        for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            raw = msg.get("data")
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            yield json.loads(raw)
    except (ConnectionError, ValueError, OSError):
        return
