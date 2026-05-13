import asyncio
import queue
import threading

import docker

_docker_client: docker.DockerClient | None = None


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
