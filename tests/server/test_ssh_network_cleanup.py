"""Tests for server-side SSH network cleanup."""

import asyncio
from unittest.mock import MagicMock, patch

from server.supervisor.adapters.docker import (
    _SSH_MANAGED_LABEL,
    _SSH_NETWORK_NAME,
    DockerWorkerFactory,
)
from server.supervisor.manager import WorkerManager
from server.supervisor.registry import WorkerRegistry


def _mock_net(
    name: str = _SSH_NETWORK_NAME, managed: str = "true", has_labels: bool = True
) -> MagicMock:
    net = MagicMock()
    net.name = name
    if has_labels:
        net.attrs = {"Labels": {_SSH_MANAGED_LABEL: managed}}
    else:
        net.attrs = {}
    return net


def _factory_with_docker(docker: MagicMock) -> DockerWorkerFactory:
    factory = MagicMock(spec=DockerWorkerFactory)
    factory._docker = docker
    return factory


# ------------------------------------------------------------------ #
# DockerWorkerFactory._remove_ssh_network
# ------------------------------------------------------------------ #


class TestRemoveSshNetwork:
    def test_removes_managed_network(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        net = _mock_net()
        docker.networks.list.return_value = [net]

        DockerWorkerFactory._remove_ssh_network(factory)

        docker.networks.list.assert_called_once_with(names=[_SSH_NETWORK_NAME])
        net.remove.assert_called_once()

    def test_ignores_partial_name_match(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        net = _mock_net(name=f"{_SSH_NETWORK_NAME}_stale")
        docker.networks.list.return_value = [net]

        DockerWorkerFactory._remove_ssh_network(factory)

        net.remove.assert_not_called()

    def test_ignores_unmanaged_network(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        net = _mock_net(managed="false")
        docker.networks.list.return_value = [net]

        DockerWorkerFactory._remove_ssh_network(factory)

        net.remove.assert_not_called()

    def test_ignores_network_without_labels(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        net = _mock_net(has_labels=False)
        docker.networks.list.return_value = [net]

        DockerWorkerFactory._remove_ssh_network(factory)

        net.remove.assert_not_called()

    def test_noop_when_no_networks(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        docker.networks.list.return_value = []

        DockerWorkerFactory._remove_ssh_network(factory)

        docker.networks.list.assert_called_once()

    def test_silently_handles_remove_failure(self) -> None:
        """Docker refuses removal if containers are still connected."""
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        net = _mock_net()
        net.remove.side_effect = RuntimeError("network has active endpoints")
        docker.networks.list.return_value = [net]

        DockerWorkerFactory._remove_ssh_network(factory)

    def test_silently_handles_list_failure(self) -> None:
        docker = MagicMock()
        factory = _factory_with_docker(docker)
        docker.networks.list.side_effect = RuntimeError("docker down")

        DockerWorkerFactory._remove_ssh_network(factory)


# ------------------------------------------------------------------ #
# WorkerManager.stop calls cleanup on factories
# ------------------------------------------------------------------ #


class TestWorkerManagerStop:
    def test_stop_cleans_up_factories(self) -> None:
        docker_factory = MagicMock()
        vastai_factory = MagicMock()
        with (
            patch(
                "server.supervisor.manager.DockerWorkerFactory.get_instance",
                return_value=docker_factory,
            ),
            patch(
                "server.supervisor.manager.VastAIWorkerFactory.get_instance",
                return_value=vastai_factory,
            ),
        ):
            manager = WorkerManager(
                MagicMock(), "missing.yaml", WorkerRegistry(), MagicMock()
            )
            manager._is_started = True

            asyncio.run(manager.stop())

        docker_factory.cleanup.assert_called_once_with()
        vastai_factory.cleanup.assert_called_once_with()
