"""`flowmesh stack --backend k8s` dispatch and Kubernetes backend behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from flowmesh.models.nodes import NodeRole
from flowmesh_cli_stack import k8s as k8s_module
from flowmesh_cli_stack import stack as stack_module
from flowmesh_cli_stack.k8s import StackBackend

ENV_FILE = Path(".env")


# ------------------------------------------------------------------ #
# Backend resolution
# ------------------------------------------------------------------ #


class TestResolveBackend:
    def test_option_wins_over_the_env_file(self) -> None:
        with patch.object(
            k8s_module, "parse_env_file", return_value={"STACK_BACKEND": "compose"}
        ):
            assert k8s_module.resolve_backend("k8s", ENV_FILE) is StackBackend.K8S

    def test_env_file_is_used_without_an_option(self) -> None:
        with patch.object(
            k8s_module, "parse_env_file", return_value={"STACK_BACKEND": "k8s"}
        ):
            assert k8s_module.resolve_backend(None, ENV_FILE) is StackBackend.K8S

    def test_compose_is_the_default(self) -> None:
        with patch.object(k8s_module, "parse_env_file", return_value={}):
            assert k8s_module.resolve_backend(None, ENV_FILE) is StackBackend.COMPOSE

    def test_unknown_backend_exits(self) -> None:
        with patch.object(k8s_module, "parse_env_file", return_value={}):
            with pytest.raises(typer.Exit):
                k8s_module.resolve_backend("nomad", ENV_FILE)


# ------------------------------------------------------------------ #
# Command dispatch
# ------------------------------------------------------------------ #


class TestDispatch:
    def test_up_routes_to_kubernetes(self) -> None:
        with (
            patch.object(
                stack_module, "resolve_backend", return_value=StackBackend.K8S
            ),
            patch.object(stack_module.k8s, "up") as up,
            patch.object(stack_module, "_compose") as compose,
        ):
            stack_module.up(env_file=ENV_FILE, image_tag=None, backend="k8s")

        up.assert_called_once()
        compose.assert_not_called()

    def test_up_leaves_the_compose_path_untouched(self) -> None:
        with (
            patch.object(
                stack_module, "resolve_backend", return_value=StackBackend.COMPOSE
            ),
            patch.object(stack_module.k8s, "up") as up,
            patch.object(stack_module, "_compose") as compose,
            patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
            patch.object(stack_module, "image_env_overrides", return_value={}),
        ):
            stack_module.up(env_file=ENV_FILE, image_tag=None, backend=None)

        up.assert_not_called()
        compose.assert_called_once()

    def test_down_routes_to_kubernetes(self) -> None:
        with (
            patch.object(
                stack_module, "resolve_backend", return_value=StackBackend.K8S
            ),
            patch.object(stack_module.k8s, "down") as down,
            patch.object(stack_module, "_compose") as compose,
            patch.object(stack_module, "drain_workers") as drain,
        ):
            stack_module.down(env_file=ENV_FILE, image_tag=None, backend="k8s")

        down.assert_called_once()
        compose.assert_not_called()
        drain.assert_not_called()

    def test_restart_routes_to_kubernetes(self) -> None:
        with (
            patch.object(
                stack_module, "resolve_backend", return_value=StackBackend.K8S
            ),
            patch.object(stack_module.k8s, "restart") as restart,
            patch.object(stack_module, "_compose") as compose,
        ):
            stack_module.restart(
                services=["server"],
                env_file=ENV_FILE,
                image_tag=None,
                pull=True,
                backend="k8s",
            )

        restart.assert_called_once()
        assert restart.call_args.kwargs["services"] == ["server"]
        compose.assert_not_called()

    def test_ps_routes_to_kubernetes(self) -> None:
        with (
            patch.object(
                stack_module, "resolve_backend", return_value=StackBackend.K8S
            ),
            patch.object(stack_module.k8s, "ps") as ps,
            patch.object(stack_module, "_compose") as compose,
        ):
            stack_module.ps(env_file=ENV_FILE, backend="k8s")

        ps.assert_called_once()
        compose.assert_not_called()


# ------------------------------------------------------------------ #
# Kubernetes lifecycle
# ------------------------------------------------------------------ #


def _stack(role: NodeRole = NodeRole.ROOT) -> MagicMock:
    target = MagicMock()
    target.apply.return_value.returncode = 0
    target.delete.return_value.returncode = 0
    target.delete_workers.return_value.returncode = 0
    target.delete_volumes.return_value.returncode = 0
    target.rollout_status.return_value.returncode = 0
    target.rollout_restart.return_value.returncode = 0
    target.logs.return_value.returncode = 0
    target.status.return_value.returncode = 0
    target.worker_status.return_value.returncode = 0
    return target


class TestKubernetesLifecycle:
    def test_up_waits_for_every_workload(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "node_role", return_value=NodeRole.ROOT),
        ):
            k8s_module.up(ENV_FILE)

        target.apply.assert_called_once()
        waited = [call.args[0] for call in target.rollout_status.call_args_list]
        assert waited == [
            "statefulset/flowmesh-redis-control",
            "statefulset/flowmesh-redis-telemetry",
            "deployment/flowmesh-server",
        ]

    def test_worker_node_waits_only_for_the_server(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "node_role", return_value=NodeRole.WORKER),
        ):
            k8s_module.up(ENV_FILE)

        waited = [call.args[0] for call in target.rollout_status.call_args_list]
        assert waited == ["deployment/flowmesh-server"]

    def test_down_drains_workers_first(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers") as drain,
        ):
            k8s_module.down(ENV_FILE)

        drain.assert_called_once()
        target.delete_workers.assert_called_once()
        target.delete.assert_called_once()

    def test_restart_of_the_server_drains_workers(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers") as drain,
        ):
            k8s_module.restart(services=["server"], env_file=ENV_FILE)

        drain.assert_called_once()
        target.rollout_restart.assert_called_once_with("deployment/flowmesh-server")

    def test_restart_of_redis_does_not_drain_workers(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers") as drain,
        ):
            k8s_module.restart(services=["redis_control"], env_file=ENV_FILE)

        drain.assert_not_called()

    def test_image_tag_is_applied_rather_than_rolled(self) -> None:
        """A rollout restart alone would leave the pod template's image unchanged."""
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers"),
        ):
            k8s_module.restart(services=["server"], env_file=ENV_FILE, image_tag="v2")

        target.apply.assert_called_once()
        target.rollout_restart.assert_not_called()

    def test_unknown_service_exits_without_acting(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers") as drain,
        ):
            with pytest.raises(typer.Exit):
                k8s_module.restart(services=["nope"], env_file=ENV_FILE)

        drain.assert_not_called()
        target.rollout_restart.assert_not_called()

    def test_clean_removes_volumes_after_teardown(self) -> None:
        target = _stack()
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "drain_workers"),
        ):
            k8s_module.clean(ENV_FILE)

        target.delete.assert_called_once()
        target.delete_volumes.assert_called_once()

    def test_ps_lists_stack_and_worker_pods(self) -> None:
        target = _stack()
        with patch.object(k8s_module, "stack", return_value=target):
            k8s_module.ps(ENV_FILE)

        target.status.assert_called_once()
        target.worker_status.assert_called_once()

    def test_failed_kubectl_call_exits_with_its_code(self) -> None:
        target = _stack()
        target.apply.return_value.returncode = 3
        with (
            patch.object(k8s_module, "stack", return_value=target),
            patch.object(k8s_module, "node_role", return_value=NodeRole.ROOT),
        ):
            with pytest.raises(typer.Exit) as exit_info:
                k8s_module.up(ENV_FILE)

        assert exit_info.value.exit_code == 3


# ------------------------------------------------------------------ #
# Environment derivation
# ------------------------------------------------------------------ #


class TestEnvDerivation:
    def test_worker_namespace_defaults_to_the_stack_namespace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("K8S_NAMESPACE", "ml-team")
        monkeypatch.delenv("K8S_WORKER_NAMESPACE", raising=False)
        monkeypatch.delenv("SERVER_WORKER_CONFIG", raising=False)

        k8s_module.apply_k8s_env(tmp_path)

        assert k8s_module.os.environ["K8S_WORKER_NAMESPACE"] == "ml-team"

    def test_explicit_worker_namespace_is_kept(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("K8S_NAMESPACE", "ml-team")
        monkeypatch.setenv("K8S_WORKER_NAMESPACE", "gpu-pool")

        k8s_module.apply_k8s_env(tmp_path)

        assert k8s_module.os.environ["K8S_WORKER_NAMESPACE"] == "gpu-pool"

    def test_distinct_worker_namespace_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("K8S_NAMESPACE", "flowmesh")
        monkeypatch.setenv("K8S_WORKER_NAMESPACE", "gpu-pool")

        k8s_module.apply_k8s_env(tmp_path)

        assert k8s_module.os.environ["K8S_WORKER_NAMESPACE_DISTINCT"] == "true"

    def test_shared_worker_namespace_is_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("K8S_NAMESPACE", "flowmesh")
        monkeypatch.delenv("K8S_WORKER_NAMESPACE", raising=False)

        k8s_module.apply_k8s_env(tmp_path)

        assert k8s_module.os.environ["K8S_WORKER_NAMESPACE_DISTINCT"] == "false"

    def test_worker_config_resolves_to_an_absolute_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("SERVER_WORKER_CONFIG", raising=False)

        k8s_module.apply_k8s_env(tmp_path)

        resolved = Path(k8s_module.os.environ["SERVER_WORKER_CONFIG"])
        assert resolved.is_absolute()
        assert resolved.name == "worker_config.yaml"
