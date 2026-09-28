import asyncio
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
    """Unbounded queue whose ``put``/``get`` run on the owning event loop.

    Other threads hand items over with
    ``asyncio.run_coroutine_threadsafe(q.put(item), loop)``.

    ``get`` must never park an executor thread. It used to be
    ``loop.run_in_executor(None, queue.Queue.get)``: a blocking get cannot be
    cancelled, so every ``StreamTasks`` call that ended while idle (a worker
    disconnecting, restarting or being recreated) left one thread of the loop's
    default executor blocked on a queue nobody would ever fill again. That pool
    is ``min(32, cpu_count + 4)`` threads; once every thread was parked, the
    executor-backed ``put`` of each new dispatch never ran either, so tasks sat
    DISPATCHED forever on workers that were connected, heartbeating and idle
    -- with nothing logged anywhere.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[T] = asyncio.Queue()

    async def put(self, item: T) -> None:
        self._q.put_nowait(item)

    async def get(self) -> T:
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()


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
