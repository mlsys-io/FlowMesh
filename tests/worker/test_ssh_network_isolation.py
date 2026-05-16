"""Tests for SSH container network isolation.

Verifies that the SSHExecutor creates an isolated Docker bridge network with
inter-container communication (ICC) disabled and attaches SSH containers to it.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.worker.factories import make_live_worker_config
from worker.config import WorkerConfig
from worker.executors.ssh_executor import SSHConfig, SSHExecutor

_SSH_NETWORK_NAME = "flowmesh_ssh_test"


def _ssh_config(image: str = "myimg:latest") -> SSHConfig:
    return SSHConfig(
        image=image,
        interactive=True,
        user="flowmesh",
        authorized_keys=[],
        command=None,
        entrypoint=None,
        ttl_sec=60.0,
        idle_sec=30.0,
        access_mode="direct",
        extra_env={},
        inputs=[],
        output=None,
        mounts=[],
        poll_interval_sec=1.0,
        stop_timeout_sec=5.0,
        cpu_limit=None,
        memory_limit_bytes=None,
        pids_limit=None,
        gpu_device_ids=[],
    )


def _worker_config(
    tmp_path: Path, ssh_network_name: str | None = _SSH_NETWORK_NAME
) -> WorkerConfig:
    return make_live_worker_config(tmp_path, ssh_network_name=ssh_network_name)


def _make_executor(
    tmp_path: Path, ssh_network_name: str | None = _SSH_NETWORK_NAME
) -> SSHExecutor:
    return SSHExecutor(_worker_config(tmp_path, ssh_network_name), lifecycle=None)


def _mock_net(
    name: str = _SSH_NETWORK_NAME, managed: str = "true", has_labels: bool = True
) -> MagicMock:
    net = MagicMock()
    net.name = name
    if has_labels:
        net.attrs = {"Labels": {"flowmesh.ssh.managed": managed}}
    else:
        net.attrs = {}
    return net


# ------------------------------------------------------------------ #
# _ensure_ssh_network
# ------------------------------------------------------------------ #


class TestEnsureSshNetwork:
    def test_creates_network_with_icc_disabled(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = []

        result = executor._ensure_ssh_network(client)

        assert result == _SSH_NETWORK_NAME
        client.networks.create.assert_called_once()
        _, kwargs = client.networks.create.call_args
        assert kwargs["driver"] == "bridge"
        assert kwargs["options"]["com.docker.network.bridge.enable_icc"] == "false"
        assert kwargs["labels"]["flowmesh.ssh.managed"] == "true"

    def test_reuses_existing_managed_network(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = [_mock_net()]

        result = executor._ensure_ssh_network(client)

        assert result == _SSH_NETWORK_NAME
        client.networks.create.assert_not_called()

    def test_does_not_reuse_unmanaged_network(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = [_mock_net(managed="false")]

        result = executor._ensure_ssh_network(client)

        assert result == _SSH_NETWORK_NAME
        client.networks.create.assert_called_once()

    def test_does_not_reuse_network_without_labels(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = [_mock_net(has_labels=False)]

        result = executor._ensure_ssh_network(client)

        assert result == _SSH_NETWORK_NAME
        client.networks.create.assert_called_once()

    def test_ignores_partial_name_match(self, tmp_path: Path) -> None:
        """Docker networks.list(names=...) does substring matching;
        we must verify the exact name."""
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = [
            _mock_net(name=f"{_SSH_NETWORK_NAME}_other")
        ]

        result = executor._ensure_ssh_network(client)

        assert result == _SSH_NETWORK_NAME
        client.networks.create.assert_called_once()

    def test_returns_none_when_ssh_network_name_not_configured(
        self, tmp_path: Path
    ) -> None:
        executor = _make_executor(tmp_path, ssh_network_name=None)
        client = MagicMock()

        result = executor._ensure_ssh_network(client)

        assert result is None
        client.networks.list.assert_not_called()
        client.networks.create.assert_not_called()

    def test_returns_none_on_create_failure(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = []
        client.networks.create.side_effect = RuntimeError("permission denied")

        result = executor._ensure_ssh_network(client)

        assert result is None

    def test_returns_none_on_list_and_create_failure(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.side_effect = RuntimeError("docker down")
        client.networks.create.side_effect = RuntimeError("docker down")

        result = executor._ensure_ssh_network(client)

        assert result is None

    def test_network_is_not_internal(self, tmp_path: Path) -> None:
        """The network must allow outbound internet (internal=False is default)."""
        executor = _make_executor(tmp_path)
        client = MagicMock()
        client.networks.list.return_value = []

        executor._ensure_ssh_network(client)

        _, kwargs = client.networks.create.call_args
        assert kwargs.get("internal") is not True


# ------------------------------------------------------------------ #
# prepare() integrates _ensure_ssh_network
# ------------------------------------------------------------------ #


class TestPrepareCreatesNetwork:
    @patch("worker.executors.ssh_executor.docker")
    def test_prepare_sets_ssh_network(
        self, mock_docker: MagicMock, tmp_path: Path
    ) -> None:
        executor = _make_executor(tmp_path)
        client = mock_docker.from_env.return_value
        client.networks.list.return_value = []

        executor.prepare()

        assert executor._ssh_network == _SSH_NETWORK_NAME
        client.networks.create.assert_called_once()

    @patch("worker.executors.ssh_executor.docker")
    def test_prepare_graceful_fallback(
        self, mock_docker: MagicMock, tmp_path: Path
    ) -> None:
        executor = _make_executor(tmp_path)
        client = mock_docker.from_env.return_value
        client.networks.list.return_value = []
        client.networks.create.side_effect = RuntimeError("no perms")

        executor.prepare()

        assert executor._ssh_network is None

    @patch("worker.executors.ssh_executor.docker")
    def test_prepare_skips_network_when_not_configured(
        self, mock_docker: MagicMock, tmp_path: Path
    ) -> None:
        executor = _make_executor(tmp_path, ssh_network_name=None)
        client = mock_docker.from_env.return_value

        executor.prepare()

        assert executor._ssh_network is None
        client.networks.list.assert_not_called()
        client.networks.create.assert_not_called()


# ------------------------------------------------------------------ #
# _build_run_kwargs attaches network
# ------------------------------------------------------------------ #


class TestBuildRunKwargsNetwork:
    def test_includes_network_when_set(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        executor._ssh_network = _SSH_NETWORK_NAME

        kwargs = executor._build_run_kwargs(
            _ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={"22/tcp": None},
            volumes=[],
            command=None,
            interactive=True,
        )

        assert kwargs["network"] == _SSH_NETWORK_NAME

    def test_omits_network_when_none(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        executor._ssh_network = None

        kwargs = executor._build_run_kwargs(
            _ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={"22/tcp": None},
            volumes=[],
            command=None,
            interactive=True,
        )

        assert "network" not in kwargs

    def test_security_opt_always_present(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        executor._ssh_network = _SSH_NETWORK_NAME

        kwargs = executor._build_run_kwargs(
            _ssh_config(),
            container_name="c",
            environment={},
            labels={},
            ports={},
            volumes=[],
            command=None,
            interactive=True,
        )

        assert kwargs["security_opt"] == ["no-new-privileges:true"]


# ------------------------------------------------------------------ #
# teardown does NOT remove network (supervisor's responsibility)
# ------------------------------------------------------------------ #


class TestTeardownSkipsNetwork:
    @patch("worker.executors.ssh_executor.docker")
    def test_teardown_does_not_touch_network(
        self, mock_docker: MagicMock, tmp_path: Path
    ) -> None:
        """Network cleanup is the supervisor's responsibility, not the worker's."""
        executor = _make_executor(tmp_path)
        client = mock_docker.from_env.return_value
        client.containers.list.return_value = []

        executor._ssh_network = _SSH_NETWORK_NAME
        executor.teardown()

        # teardown should only list containers (for stopping), never networks.
        for call in client.networks.method_calls:
            assert "remove" not in str(call)
