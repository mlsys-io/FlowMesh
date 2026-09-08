"""Kubernetes worker provider: pod construction, lifecycle, and hardware probe."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from server.hooks import PrincipalContext
from server.supervisor.adapters import kubernetes as k8s_adapter
from server.supervisor.adapters.base import WorkerTokenType, WorkerType
from server.supervisor.adapters.kubernetes import (
    MANAGED_LABEL,
    NODE_ALIAS_LABEL,
    WORKER_NAME_LABEL,
    KubernetesWorkerAdapter,
    KubernetesWorkerConfig,
    KubernetesWorkerFactory,
    _secret_field_names,
    sanitize_label_value,
    sanitize_object_name,
)
from server.supervisor.resource_manager import GpuArch
from server.supervisor.schemas import WorkerStatus

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _token(value: str) -> WorkerTokenType:
    return WorkerTokenType(value)


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-test",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


def _adapter(
    core: MagicMock | None = None, **config_kwargs: Any
) -> KubernetesWorkerAdapter:
    return KubernetesWorkerAdapter(
        token=_token("tok-123"),
        name="worker-1",
        pod_name="fm-worker-1",
        config=KubernetesWorkerConfig(**config_kwargs),
        core_api=core or MagicMock(),
        node_alias="flowmesh_node",
        owner=_principal(),
    )


def _api_error(status: int) -> ApiException:
    return ApiException(status=status, reason="test")


def _container(manifest: dict[str, Any]) -> dict[str, Any]:
    containers: list[dict[str, Any]] = manifest["spec"]["containers"]
    return containers[0]


def _env_names(manifest: dict[str, Any]) -> set[str]:
    return {entry["name"] for entry in _container(manifest).get("env", [])}


def _node(allocatable: dict[str, str], labels: dict[str, str]) -> MagicMock:
    node = MagicMock()
    node.status.allocatable = allocatable
    node.metadata.labels = labels
    return node


# ------------------------------------------------------------------ #
# Object naming
# ------------------------------------------------------------------ #


class TestObjectNaming:
    def test_underscores_and_case_are_normalized(self) -> None:
        assert sanitize_object_name("flowmesh_node-Worker_CPU_1") == (
            "flowmesh-node-worker-cpu-1"
        )

    def test_long_names_are_truncated_with_a_digest(self) -> None:
        name = sanitize_object_name("x" * 200)
        assert len(name) <= 63
        assert name != sanitize_object_name("y" * 200)

    def test_distinct_long_names_do_not_collide(self) -> None:
        base = "flowmesh-server-worker-gpu-" + "a" * 60
        assert sanitize_object_name(base + "-one") != sanitize_object_name(
            base + "-two"
        )

    def test_name_starting_with_a_separator_is_replaced(self) -> None:
        assert sanitize_object_name("___").startswith("w-")

    def test_label_values_keep_underscores(self) -> None:
        assert sanitize_label_value("flowmesh_node") == "flowmesh_node"


# ------------------------------------------------------------------ #
# Environment split
# ------------------------------------------------------------------ #


class TestEnvironmentSplit:
    def test_credentials_go_to_the_secret_and_not_the_pod(self) -> None:
        adapter = _adapter()
        inline, secret = adapter._split_environment()

        assert "WORKER_TOKEN" in secret
        assert "FLOWMESH_API_KEY" in secret
        assert "WORKER_TOKEN" not in inline
        assert "FLOWMESH_API_KEY" not in inline

    def test_secret_str_fields_are_all_treated_as_secret(self) -> None:
        adapter = _adapter()
        _, secret = adapter._split_environment()

        for field in _secret_field_names(KubernetesWorkerConfig):
            assert field.upper() in secret

    def test_derived_secret_keys_exist_in_the_worker_environment(self) -> None:
        """Guards the field-name -> env-key assumption behind the split."""
        adapter = _adapter()
        base_env = adapter._base_environment()

        for field in _secret_field_names(KubernetesWorkerConfig):
            assert field.upper() in base_env

    def test_non_secret_values_stay_inline(self) -> None:
        adapter = _adapter()
        inline, secret = adapter._split_environment()

        assert "SUPERVISOR_GRPC_TARGET" in inline
        assert "SUPERVISOR_GRPC_TLS_CA_B64" in inline
        assert not set(inline) & set(secret)

    def test_results_dir_points_at_the_mount_path(self) -> None:
        adapter = _adapter(results_mount_path="/mnt/results")
        assert adapter._base_environment()["RESULTS_DIR"] == "/mnt/results"

    def test_flowmesh_url_defaults_to_the_in_cluster_service(self) -> None:
        adapter = _adapter()
        url = adapter._base_environment()["FLOWMESH_BASE_URL"]
        assert url.startswith("http://flowmesh-server.")
        assert ".svc." in url


# ------------------------------------------------------------------ #
# Pod manifest
# ------------------------------------------------------------------ #


class TestPodManifest:
    def test_labels_identify_the_managing_node(self) -> None:
        manifest = _adapter().build_pod_manifest()
        labels = manifest["metadata"]["labels"]

        assert labels[MANAGED_LABEL] == "true"
        assert labels[NODE_ALIAS_LABEL] == "flowmesh_node"
        assert labels[WORKER_NAME_LABEL] == "worker-1"

    def test_reaping_selector_matches_the_pod_labels(self) -> None:
        """The reaper keys on the node alias, which survives a restart."""
        labels = _adapter().build_pod_manifest()["metadata"]["labels"]

        assert labels[MANAGED_LABEL] == "true"
        assert NODE_ALIAS_LABEL in labels

    def test_pod_restarts_in_place(self) -> None:
        manifest = _adapter().build_pod_manifest()
        assert manifest["spec"]["restartPolicy"] == "Always"

    def test_secret_is_consumed_via_env_from(self) -> None:
        adapter = _adapter()
        manifest = adapter.build_pod_manifest()
        env_from = _container(manifest)["envFrom"]

        assert env_from == [{"secretRef": {"name": adapter.secret_name}}]

    def test_cpu_worker_requests_no_gpu(self) -> None:
        manifest = _adapter().build_pod_manifest()
        limits = _container(manifest).get("resources", {}).get("limits", {})
        assert "nvidia.com/gpu" not in limits

    def test_gpu_worker_requests_the_gpu_resource(self) -> None:
        manifest = _adapter(
            worker_type=WorkerType.GPU, gpu_count=2
        ).build_pod_manifest()
        limits = _container(manifest)["resources"]["limits"]
        assert limits["nvidia.com/gpu"] == "2"

    def test_gpu_resource_name_is_configurable(self) -> None:
        manifest = _adapter(
            worker_type=WorkerType.GPU, gpu_resource_name="amd.com/gpu"
        ).build_pod_manifest()
        assert "amd.com/gpu" in _container(manifest)["resources"]["limits"]

    def test_device_indices_are_never_pinned_server_side(self) -> None:
        """The device plugin owns assignment; a server-set index would fight it."""
        manifest = _adapter(worker_type=WorkerType.GPU).build_pod_manifest()
        assert "CUDA_VISIBLE_DEVICES" not in _env_names(manifest)

    def test_results_default_to_an_ephemeral_volume(self) -> None:
        manifest = _adapter().build_pod_manifest()
        volumes = {v["name"]: v for v in manifest["spec"]["volumes"]}
        assert volumes["results"]["emptyDir"] == {}

    def test_results_pvc_is_mounted_when_configured(self) -> None:
        manifest = _adapter(results_pvc="fm-results").build_pod_manifest()
        volumes = {v["name"]: v for v in manifest["spec"]["volumes"]}
        assert volumes["results"]["persistentVolumeClaim"]["claimName"] == "fm-results"

    def test_shm_volume_is_added_when_sized(self) -> None:
        manifest = _adapter(shm_size="2Gi").build_pod_manifest()
        volumes = {v["name"]: v for v in manifest["spec"]["volumes"]}
        mounts = {m["name"]: m for m in _container(manifest)["volumeMounts"]}

        assert volumes["dshm"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "2Gi"}
        assert mounts["dshm"]["mountPath"] == "/dev/shm"

    def test_no_shm_volume_by_default(self) -> None:
        manifest = _adapter().build_pod_manifest()
        assert "dshm" not in {v["name"] for v in manifest["spec"]["volumes"]}

    def test_scheduling_hints_are_carried_through(self) -> None:
        manifest = _adapter(
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            service_account_name="fm-worker",
            image_pull_secrets=["regcred"],
            runtime_class_name="nvidia",
            priority_class_name="high",
        ).build_pod_manifest()
        spec = manifest["spec"]

        assert spec["nodeSelector"] == {"gpu": "true"}
        assert spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert spec["serviceAccountName"] == "fm-worker"
        assert spec["imagePullSecrets"] == [{"name": "regcred"}]
        assert spec["runtimeClassName"] == "nvidia"
        assert spec["priorityClassName"] == "high"

    def test_unset_optional_fields_are_omitted(self) -> None:
        spec = _adapter().build_pod_manifest()["spec"]
        for key in ("nodeSelector", "tolerations", "runtimeClassName"):
            assert key not in spec

    def test_pod_overrides_deep_merge_mappings(self) -> None:
        manifest = _adapter(
            pod_overrides={"metadata": {"annotations": {"team": "ml"}}}
        ).build_pod_manifest()

        assert manifest["metadata"]["annotations"] == {"team": "ml"}
        assert manifest["metadata"]["name"] == "fm-worker-1"

    def test_pod_overrides_replace_sequences(self) -> None:
        manifest = _adapter(
            pod_overrides={"spec": {"tolerations": [{"key": "override"}]}}
        ).build_pod_manifest()

        assert manifest["spec"]["tolerations"] == [{"key": "override"}]

    def test_secret_manifest_carries_only_credentials(self) -> None:
        adapter = _adapter()
        secret = adapter.build_secret_manifest()
        _, expected = adapter._split_environment()

        assert secret["kind"] == "Secret"
        assert secret["stringData"] == expected


# ------------------------------------------------------------------ #
# Lifecycle
# ------------------------------------------------------------------ #


class TestLifecycle:
    def test_start_creates_the_secret_before_the_pod(self) -> None:
        core = MagicMock()
        core.read_namespaced_pod.side_effect = _api_error(404)
        adapter = _adapter(core)

        assert asyncio.run(adapter.start()) is True
        core.create_namespaced_secret.assert_called_once()
        core.create_namespaced_pod.assert_called_once()

    def test_start_adopts_a_running_pod(self) -> None:
        core = MagicMock()
        core.read_namespaced_pod.return_value.status.phase = "Running"
        adapter = _adapter(core)

        assert asyncio.run(adapter.start()) is True
        core.create_namespaced_pod.assert_not_called()

    def test_start_replaces_a_terminal_pod(self) -> None:
        core = MagicMock()
        terminal = MagicMock()
        terminal.status.phase = "Failed"
        core.read_namespaced_pod.side_effect = [terminal, _api_error(404)]
        adapter = _adapter(core)

        assert asyncio.run(adapter.start()) is True
        core.delete_namespaced_pod.assert_called_once()
        core.create_namespaced_pod.assert_called_once()

    def test_replacing_a_pod_waits_for_the_name_to_free(self) -> None:
        """Deletion is asynchronous; reusing the name too early collides."""
        core = MagicMock()
        terminal = MagicMock()
        terminal.status.phase = "Failed"
        terminating = MagicMock()
        terminating.status.phase = "Failed"
        core.read_namespaced_pod.side_effect = [
            terminal,
            terminating,
            _api_error(404),
        ]
        adapter = _adapter(core)

        with patch.object(k8s_adapter, "_DELETE_POLL_SEC", 0):
            assert asyncio.run(adapter.start()) is True

        assert core.read_namespaced_pod.call_count == 3
        core.create_namespaced_pod.assert_called_once()
        assert core.delete_namespaced_pod.call_args.kwargs["grace_period_seconds"] == 0

    def test_replacement_gives_up_when_the_pod_never_goes_away(self) -> None:
        core = MagicMock()
        terminal = MagicMock()
        terminal.status.phase = "Failed"
        core.read_namespaced_pod.return_value = terminal
        adapter = _adapter(core)

        with patch.object(k8s_adapter, "_DELETE_TIMEOUT_SEC", 0):
            assert asyncio.run(adapter.start()) is False

        core.create_namespaced_pod.assert_not_called()

    def test_failed_pod_creation_removes_the_orphaned_secret(self) -> None:
        core = MagicMock()
        core.read_namespaced_pod.side_effect = _api_error(404)
        core.create_namespaced_pod.side_effect = _api_error(422)
        adapter = _adapter(core)

        assert asyncio.run(adapter.start()) is False
        core.delete_namespaced_secret.assert_called_once()
        assert adapter.status is WorkerStatus.STOPPED

    def test_existing_secret_is_replaced(self) -> None:
        core = MagicMock()
        core.read_namespaced_pod.side_effect = _api_error(404)
        core.create_namespaced_secret.side_effect = _api_error(409)
        adapter = _adapter(core)

        assert asyncio.run(adapter.start()) is True
        core.replace_namespaced_secret.assert_called_once()

    def test_stop_deletes_pod_and_secret(self) -> None:
        core = MagicMock()
        adapter = _adapter(core)
        adapter.set_status(WorkerStatus.RUNNING)

        assert asyncio.run(adapter.stop()) is True
        core.delete_namespaced_pod.assert_called_once()
        core.delete_namespaced_secret.assert_called_once()

    def test_stop_tolerates_an_already_deleted_pod(self) -> None:
        core = MagicMock()
        core.delete_namespaced_pod.side_effect = _api_error(404)
        core.delete_namespaced_secret.side_effect = _api_error(404)
        adapter = _adapter(core)
        adapter.set_status(WorkerStatus.RUNNING)

        assert asyncio.run(adapter.stop()) is True

    def test_stop_reports_failure_and_restores_status(self) -> None:
        core = MagicMock()
        core.delete_namespaced_pod.side_effect = _api_error(500)
        adapter = _adapter(core)
        adapter.set_status(WorkerStatus.RUNNING)

        assert asyncio.run(adapter.stop()) is False
        assert adapter.status is WorkerStatus.RUNNING

    def test_stopping_a_stopped_worker_is_a_no_op(self) -> None:
        core = MagicMock()
        adapter = _adapter(core)

        assert asyncio.run(adapter.stop()) is True
        core.delete_namespaced_pod.assert_not_called()


# ------------------------------------------------------------------ #
# Hardware probe
# ------------------------------------------------------------------ #


class TestHardwareProbe:
    def test_probe_degrades_without_node_permission(self) -> None:
        core = MagicMock()
        core.list_node.side_effect = _api_error(403)
        adapter = _adapter(core)

        asyncio.run(adapter.prepare())
        assert adapter.get_info().hardware is None

    def test_probe_reads_allocatable_capacity(self) -> None:
        core = MagicMock()
        core.list_node.return_value.items = [
            _node({"cpu": "32", "memory": "65536Ki"}, {})
        ]
        adapter = _adapter(core)

        asyncio.run(adapter.prepare())
        hardware = adapter.get_info().hardware

        assert hardware is not None
        assert hardware.cpu.logical_cores == 32
        assert hardware.memory.total_bytes == 65536 * 1024

    def test_probe_parses_millicore_quantities(self) -> None:
        core = MagicMock()
        core.list_node.return_value.items = [_node({"cpu": "7800m"}, {})]
        adapter = _adapter(core)

        asyncio.run(adapter.prepare())
        hardware = adapter.get_info().hardware

        assert hardware is not None
        assert hardware.cpu.logical_cores == 7

    def test_probe_derives_gpu_arch_from_node_labels(self) -> None:
        core = MagicMock()
        core.list_node.return_value.items = [
            _node({"cpu": "8"}, {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"})
        ]
        adapter = _adapter(core, worker_type=WorkerType.GPU, gpu_count=2)

        asyncio.run(adapter.prepare())
        hardware = adapter.get_info().hardware

        assert hardware is not None
        assert hardware.gpu.gpu_arch == GpuArch.HOPPER.value
        assert len(hardware.gpu.devices) == 2

    def test_probe_without_matching_nodes_returns_nothing(self) -> None:
        core = MagicMock()
        core.list_node.return_value.items = []
        adapter = _adapter(core)

        asyncio.run(adapter.prepare())
        assert adapter.get_info().hardware is None

    def test_probe_filters_by_node_selector(self) -> None:
        core = MagicMock()
        core.list_node.return_value.items = []
        adapter = _adapter(core, node_selector={"gpu": "true", "zone": "a"})

        asyncio.run(adapter.prepare())
        selector = core.list_node.call_args.kwargs["label_selector"]

        assert set(selector.split(",")) == {"gpu=true", "zone=a"}


# ------------------------------------------------------------------ #
# Image selection
# ------------------------------------------------------------------ #


class TestImageSelection:
    def test_cpu_worker_uses_the_cpu_image(self) -> None:
        assert _adapter().get_image_name().endswith("-cpu")

    def test_gpu_worker_uses_the_gpu_image(self) -> None:
        adapter = _adapter(worker_type=WorkerType.GPU)
        assert adapter.get_image_name().endswith("-gpu")


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #


class TestFactory:
    def _factory(self, core: MagicMock) -> KubernetesWorkerFactory:
        factory = KubernetesWorkerFactory(_principal())
        factory._core = core
        return factory

    def test_orphaned_pods_are_reaped_once(self) -> None:
        core = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "fm-stale"
        core.list_namespaced_pod.return_value.items = [pod]
        factory = self._factory(core)

        factory.create_worker(_token("t1"), KubernetesWorkerConfig())
        factory.create_worker(_token("t2"), KubernetesWorkerConfig())

        core.list_namespaced_pod.assert_called_once()
        core.delete_namespaced_pod.assert_called_once_with(
            name="fm-stale", namespace="default"
        )

    def test_reaping_selects_this_node_only(self) -> None:
        core = MagicMock()
        core.list_namespaced_pod.return_value.items = []
        factory = self._factory(core)

        factory.create_worker(_token("t1"), KubernetesWorkerConfig())
        selector = core.list_namespaced_pod.call_args.kwargs["label_selector"]

        assert f"{MANAGED_LABEL}=true" in selector
        assert NODE_ALIAS_LABEL in selector

    def test_reaping_survives_a_listing_failure(self) -> None:
        core = MagicMock()
        core.list_namespaced_pod.side_effect = _api_error(403)
        factory = self._factory(core)

        worker = factory.create_worker(_token("t1"), KubernetesWorkerConfig())
        assert worker.name

    def test_pod_names_are_api_safe(self) -> None:
        core = MagicMock()
        core.list_namespaced_pod.return_value.items = []
        factory = self._factory(core)

        worker = factory.create_worker(
            _token("t1"), KubernetesWorkerConfig(worker_alias="Train_Worker_01")
        )

        assert worker.pod_name == sanitize_object_name(worker.pod_name)

    def test_worker_names_increment_per_type(self) -> None:
        core = MagicMock()
        core.list_namespaced_pod.return_value.items = []
        factory = self._factory(core)

        first = factory.create_worker(_token("t1"), KubernetesWorkerConfig())
        second = factory.create_worker(_token("t2"), KubernetesWorkerConfig())

        assert first.name != second.name

    def test_cleanup_closes_the_api_client(self) -> None:
        factory = KubernetesWorkerFactory(_principal())
        api_client = MagicMock()
        factory._api_client = api_client
        factory._core = MagicMock()

        factory.cleanup()

        api_client.close.assert_called_once_with()
        assert factory._core is None

    def test_client_falls_back_to_kubeconfig_outside_a_cluster(self) -> None:
        from kubernetes import config as k8s_config

        factory = KubernetesWorkerFactory(_principal())
        with (
            patch.object(
                k8s_config,
                "load_incluster_config",
                side_effect=k8s_config.ConfigException("not in cluster"),
            ),
            patch.object(k8s_config, "load_kube_config") as load_kube_config,
        ):
            factory.core_api()

        load_kube_config.assert_called_once()
        factory.cleanup()

    def test_destroy_rejects_a_foreign_worker(self) -> None:
        factory = KubernetesWorkerFactory(_principal())
        with pytest.raises(ValueError, match="Invalid worker type"):
            factory.destroy_worker(MagicMock())
