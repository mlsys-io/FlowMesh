"""Tests for the shared Docker daemon connection helpers."""

import pytest

from worker.executors.utils import docker as docker_utils
from worker.executors.utils.docker import (
    DockerUnavailableError,
    docker_available,
    docker_client,
)


def test_docker_client_raises_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        @staticmethod
        def from_env() -> None:
            raise OSError("no such file or directory")

    monkeypatch.setattr(docker_utils, "_HAS_DOCKER", True)
    monkeypatch.setattr(docker_utils, "docker", _Boom)
    with pytest.raises(DockerUnavailableError):
        docker_client()


def test_docker_client_raises_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_utils, "_HAS_DOCKER", False)
    with pytest.raises(DockerUnavailableError):
        docker_client()


def test_docker_available_false_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise DockerUnavailableError("unreachable")

    monkeypatch.setattr(docker_utils, "docker_client", _raise)
    assert docker_available() is False


def test_docker_available_true_when_ping_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def ping(self) -> bool:
            return True

        def close(self) -> None:
            return None

    monkeypatch.setattr(docker_utils, "docker_client", lambda: _Client())
    assert docker_available() is True
