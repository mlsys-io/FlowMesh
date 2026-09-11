"""Kubernetes manifest rendering: substitution, directives, and shipped assets."""

from pathlib import Path

import pytest
import yaml
from flowmesh_cli.core.assets import asset_path
from flowmesh_stack.manifests import (
    ManifestError,
    evaluate_condition,
    render_documents,
    render_manifests,
    substitute,
)

MANIFEST_ASSETS = (
    "00-namespace.yaml",
    "10-rbac.yaml",
    "20-redis.yaml",
    "30-server.yaml",
)


def _asset_paths() -> list[Path]:
    return [
        asset_path("flowmesh_cli_stack.assets", "k8s", name) for name in MANIFEST_ASSETS
    ]


def _base_env(**overrides: str) -> dict[str, str]:
    env = {
        "K8S_NAMESPACE": "flowmesh",
        "K8S_WORKER_NAMESPACE": "flowmesh",
        "NODE_ROLE": "root",
    }
    env.update(overrides)
    return env


def _by_kind(documents: list[dict]) -> dict[tuple[str, str], dict]:
    return {(doc["kind"], doc["metadata"]["name"]): doc for doc in documents}


# ------------------------------------------------------------------ #
# Substitution
# ------------------------------------------------------------------ #


class TestSubstitution:
    def test_value_is_expanded(self) -> None:
        assert substitute("ns/${NAME}", {"NAME": "flowmesh"}) == "ns/flowmesh"

    def test_default_is_used_when_unset(self) -> None:
        assert substitute("${NAME:-fallback}", {}) == "fallback"

    def test_set_value_beats_the_default(self) -> None:
        assert substitute("${NAME:-fallback}", {"NAME": "real"}) == "real"

    def test_empty_default_renders_empty(self) -> None:
        assert substitute("${NAME:-}", {}) == ""

    def test_multiple_references_in_one_value(self) -> None:
        rendered = substitute("${A}:${B}", {"A": "host", "B": "8000"})
        assert rendered == "host:8000"

    def test_missing_value_without_a_default_is_an_error(self) -> None:
        with pytest.raises(ManifestError, match="NAME is not set"):
            substitute("${NAME}", {})


# ------------------------------------------------------------------ #
# Conditions
# ------------------------------------------------------------------ #


class TestConditions:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, value: str) -> None:
        assert evaluate_condition("FLAG", {"FLAG": value}) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsey_values(self, value: str) -> None:
        assert evaluate_condition("FLAG", {"FLAG": value}) is False

    def test_unset_is_falsey(self) -> None:
        assert evaluate_condition("FLAG", {}) is False

    def test_negation(self) -> None:
        assert evaluate_condition("!FLAG", {}) is True
        assert evaluate_condition("!FLAG", {"FLAG": "1"}) is False

    def test_equality(self) -> None:
        assert evaluate_condition("ROLE==root", {"ROLE": "root"}) is True
        assert evaluate_condition("ROLE==root", {"ROLE": "worker"}) is False

    def test_empty_expression_is_an_error(self) -> None:
        with pytest.raises(ManifestError, match="Empty"):
            evaluate_condition("  ", {})


# ------------------------------------------------------------------ #
# Directives
# ------------------------------------------------------------------ #


class TestDirectives:
    def test_falsey_document_leaves_the_stream(self) -> None:
        source = """
        x-flowmesh-when: ENABLED
        kind: ConfigMap
        """
        assert render_documents(source, {}) == []

    def test_truthy_document_drops_only_the_marker(self) -> None:
        source = """
        x-flowmesh-when: ENABLED
        kind: ConfigMap
        """
        assert render_documents(source, {"ENABLED": "1"}) == [{"kind": "ConfigMap"}]

    def test_falsey_sequence_entry_leaves_the_sequence(self) -> None:
        source = """
        kind: Pod
        items:
          - name: keep
          - x-flowmesh-when: TLS
            name: drop
        """
        document = render_documents(source, {})[0]
        assert document["items"] == [{"name": "keep"}]

    def test_falsey_mapping_value_takes_its_key(self) -> None:
        source = """
        kind: Pod
        spec:
          storageClassName:
            x-flowmesh-when: CLASS
            x-flowmesh-value: ${CLASS:-}
        """
        document = render_documents(source, {})[0]
        assert "storageClassName" not in document["spec"]

    def test_conditional_scalar_renders_bare(self) -> None:
        source = """
        kind: Pod
        args:
          - x-flowmesh-when: ACL
            x-flowmesh-value: --aclfile
        """
        document = render_documents(source, {"ACL": "1"})[0]
        assert document["args"] == ["--aclfile"]

    def test_int_directive_yields_an_integer(self) -> None:
        source = """
        kind: Service
        port:
          x-flowmesh-int: ${PORT:-8000}
        """
        document = render_documents(source, {})[0]
        assert document["port"] == 8000
        assert isinstance(document["port"], int)

    def test_int_directive_rejects_non_numeric_values(self) -> None:
        source = """
        kind: Service
        port:
          x-flowmesh-int: ${PORT:-http}
        """
        with pytest.raises(ManifestError, match="expected an integer"):
            render_documents(source, {})

    def test_file_directive_inlines_contents(self, tmp_path: Path) -> None:
        config = tmp_path / "worker_config.yaml"
        config.write_text("workers: []\n")
        source = """
        kind: ConfigMap
        data:
          worker_config.yaml:
            x-flowmesh-file: SERVER_WORKER_CONFIG
        """
        document = render_documents(source, {"SERVER_WORKER_CONFIG": str(config)})[0]
        assert document["data"]["worker_config.yaml"] == "workers: []\n"

    def test_file_directive_requires_the_variable(self) -> None:
        source = """
        kind: ConfigMap
        data:
          x-flowmesh-file: SERVER_WORKER_CONFIG
        """
        with pytest.raises(ManifestError, match="readable file"):
            render_documents(source, {})

    def test_env_values_directive_uses_the_env_file_only(self) -> None:
        source = """
        kind: Secret
        stringData:
          x-flowmesh-env-values: true
        """
        document = render_documents(
            source, {"PROCESS_ONLY": "leaked"}, {"FROM_FILE": "kept"}
        )[0]
        assert document["stringData"] == {"FROM_FILE": "kept"}


# ------------------------------------------------------------------ #
# Rendering assets
# ------------------------------------------------------------------ #


class TestRenderManifests:
    def test_document_order_is_preserved(self, tmp_path: Path) -> None:
        first = tmp_path / "a.yaml"
        first.write_text("kind: Namespace\n---\nkind: Role\n")
        second = tmp_path / "b.yaml"
        second.write_text("kind: Deployment\n")

        documents = list(yaml.safe_load_all(render_manifests([first, second], {})))
        assert [doc["kind"] for doc in documents] == [
            "Namespace",
            "Role",
            "Deployment",
        ]

    def test_empty_output_is_an_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("x-flowmesh-when: NEVER\nkind: ConfigMap\n")
        with pytest.raises(ManifestError, match="No manifests"):
            render_manifests([empty], {})

    def test_unreadable_asset_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="Failed to read"):
            render_manifests([tmp_path / "missing.yaml"], {})


# ------------------------------------------------------------------ #
# Shipped assets
# ------------------------------------------------------------------ #


class TestShippedAssets:
    def _documents(self, **overrides: str) -> list[dict]:
        env = _base_env(**overrides)
        env.setdefault(
            "SERVER_WORKER_CONFIG",
            asset_path(
                "flowmesh_cli_stack.assets", "k8s", "worker_config.k8s.yaml"
            ).as_posix(),
        )
        return list(yaml.safe_load_all(render_manifests(_asset_paths(), env, env)))

    def test_namespace_is_applied_first(self) -> None:
        assert self._documents()[0]["kind"] == "Namespace"

    def test_only_one_namespace_when_workers_share_it(self) -> None:
        namespaces = [
            doc["metadata"]["name"]
            for doc in self._documents()
            if doc["kind"] == "Namespace"
        ]
        assert namespaces == ["flowmesh"]

    def test_a_separate_worker_namespace_is_created(self) -> None:
        """RBAC is bound in the worker namespace, so it has to exist."""
        namespaces = [
            doc["metadata"]["name"]
            for doc in self._documents(
                K8S_WORKER_NAMESPACE="gpu-pool",
                K8S_WORKER_NAMESPACE_DISTINCT="true",
            )
            if doc["kind"] == "Namespace"
        ]
        assert namespaces == ["flowmesh", "gpu-pool"]

    def test_root_node_ships_both_redis_workloads(self) -> None:
        names = {
            doc["metadata"]["name"]
            for doc in self._documents()
            if doc["kind"] == "StatefulSet"
        }
        assert names == {"flowmesh-redis-control", "flowmesh-redis-telemetry"}

    def test_worker_node_ships_no_redis(self) -> None:
        documents = self._documents(NODE_ROLE="worker")
        assert not [doc for doc in documents if doc["kind"] == "StatefulSet"]
        assert [doc for doc in documents if doc["kind"] == "Deployment"]

    def test_server_runs_a_single_replica(self) -> None:
        deployment = _by_kind(self._documents())[("Deployment", "flowmesh-server")]
        assert deployment["spec"]["replicas"] == 1
        assert deployment["spec"]["strategy"]["type"] == "Recreate"

    def test_node_rbac_is_opt_in(self) -> None:
        kinds = {doc["kind"] for doc in self._documents()}
        assert "ClusterRole" not in kinds

        enabled = {doc["kind"] for doc in self._documents(K8S_ENABLE_NODE_RBAC="true")}
        assert {"ClusterRole", "ClusterRoleBinding"} <= enabled

    def test_service_ports_are_integers(self) -> None:
        service = _by_kind(self._documents())[("Service", "flowmesh-server")]
        assert isinstance(service["spec"]["ports"][0]["port"], int)

    def test_storage_class_is_omitted_when_unset(self) -> None:
        claim = _by_kind(self._documents())[
            ("PersistentVolumeClaim", "flowmesh-results")
        ]
        assert "storageClassName" not in claim["spec"]

    def test_storage_class_is_set_when_configured(self) -> None:
        claim = _by_kind(self._documents(K8S_STORAGE_CLASS="fast"))[
            ("PersistentVolumeClaim", "flowmesh-results")
        ]
        assert claim["spec"]["storageClassName"] == "fast"

    def test_redis_volumes_are_labelled_for_cleanup(self) -> None:
        """`clean` selects volumes by label; controller-made claims need it too."""
        for name in ("flowmesh-redis-control", "flowmesh-redis-telemetry"):
            statefulset = _by_kind(self._documents())[("StatefulSet", name)]
            claim = statefulset["spec"]["volumeClaimTemplates"][0]
            assert (
                claim["metadata"]["labels"]["app.kubernetes.io/part-of"] == "flowmesh"
            )

    def test_redis_keeps_the_pubsub_buffer_limit(self) -> None:
        statefulset = _by_kind(self._documents())[
            ("StatefulSet", "flowmesh-redis-control")
        ]
        args = statefulset["spec"]["template"]["spec"]["containers"][0]["args"]
        assert "--client-output-buffer-limit" in args

    def test_redis_acl_is_wired_only_when_enabled(self) -> None:
        without = _by_kind(self._documents())[("StatefulSet", "flowmesh-redis-control")]
        assert (
            "--aclfile"
            not in without["spec"]["template"]["spec"]["containers"][0]["args"]
        )

        with_acl = _by_kind(self._documents(REDIS_ACL_ENABLED="true"))[
            ("StatefulSet", "flowmesh-redis-control")
        ]
        spec = with_acl["spec"]["template"]["spec"]
        assert "--aclfile" in spec["containers"][0]["args"]
        assert [volume["name"] for volume in spec["volumes"]] == ["acl"]

    def test_tls_secrets_are_mounted_only_when_named(self) -> None:
        deployment = _by_kind(self._documents(SERVER_GRPC_TLS_SECRET="fm-tls"))[
            ("Deployment", "flowmesh-server")
        ]
        volumes = {v["name"] for v in deployment["spec"]["template"]["spec"]["volumes"]}
        assert "server-tls" in volumes

        default = _by_kind(self._documents())[("Deployment", "flowmesh-server")]
        default_volumes = {
            v["name"] for v in default["spec"]["template"]["spec"]["volumes"]
        }
        assert "server-tls" not in default_volumes

    def test_server_receives_the_namespace_from_the_downward_api(self) -> None:
        deployment = _by_kind(self._documents())[("Deployment", "flowmesh-server")]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        namespace_env = next(
            entry for entry in container["env"] if entry["name"] == "POD_NAMESPACE"
        )
        assert (
            namespace_env["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.namespace"
        )

    def test_worker_config_is_carried_in_a_config_map(self) -> None:
        config_map = _by_kind(self._documents())[
            ("ConfigMap", "flowmesh-worker-config")
        ]
        assert "provider: kubernetes" in config_map["data"]["worker_config.yaml"]

    def test_env_file_values_reach_the_server_secret(self) -> None:
        secret = _by_kind(self._documents(FLOWMESH_API_KEY="secret-key"))[
            ("Secret", "flowmesh-server-env")
        ]
        assert secret["stringData"]["FLOWMESH_API_KEY"] == "secret-key"
