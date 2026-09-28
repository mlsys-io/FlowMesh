import threading

import docker

_docker_client: docker.DockerClient | None = None


def get_docker_client() -> docker.DockerClient:
    global _docker_client
    if _docker_client is not None:
        return _docker_client

    _docker_client = docker.from_env()
    return _docker_client


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
