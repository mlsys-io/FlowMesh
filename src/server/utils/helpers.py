import asyncio
import json
import logging
import queue
import threading
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
