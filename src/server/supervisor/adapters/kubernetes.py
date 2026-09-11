import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, get_args

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from pydantic import Field, SecretStr

from shared.utils import parse_mem_to_bytes

from ... import env
from ...hooks import PrincipalContext
from ...schemas.node import CPUInfo, GpuInfo, GpuPlatformInfo, MemoryInfo
from ..resource_manager import GpuArch
from ..schemas import WorkerHardware, WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
    WorkerType,
)
from .utils import get_worker_image_name

PROVIDER_NAME = "kubernetes"

MANAGED_LABEL = "flowmesh.io/managed"
NODE_ALIAS_LABEL = "flowmesh.io/node-alias"
WORKER_NAME_LABEL = "flowmesh.io/worker-name"

_GPU_PRODUCT_LABEL = "nvidia.com/gpu.product"
_RFC1123_MAX_LEN = 63
_RFC1123_INVALID_RE = re.compile(r"[^a-z0-9-]+")
_LABEL_VALUE_INVALID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_LABEL_VALUE_MAX_LEN = 63
_CPU_MILLI_RE = re.compile(r"^([0-9]+)m$")
_DELETE_TIMEOUT_SEC = 120.0
_DELETE_POLL_SEC = 0.5

logger = logging.getLogger("supervisor")


def _short_digest(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def sanitize_object_name(name: str, maxlen: int = _RFC1123_MAX_LEN) -> str:
    """Return an RFC 1123 DNS label derived from ``name``.

    Kubernetes object names are far stricter than Docker container names —
    lowercase alphanumerics and dashes only — so names that are valid
    elsewhere in FlowMesh (``flowmesh_node``, alias-derived names carrying
    underscores) are rejected by the API. A truncated name carries a digest of
    the original so two long names cannot collapse onto one object.
    """
    sanitized = _RFC1123_INVALID_RE.sub("-", name.strip().lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    if len(sanitized) > maxlen:
        head = sanitized[: maxlen - 9].rstrip("-")
        sanitized = f"{head}-{_short_digest(name)}"
    if not sanitized or not sanitized[0].isalnum():
        sanitized = f"w-{_short_digest(name)}"
    return sanitized


def sanitize_label_value(value: str) -> str:
    """Return a value accepted by the Kubernetes label-value grammar."""
    sanitized = _LABEL_VALUE_INVALID_RE.sub("-", value.strip())[:_LABEL_VALUE_MAX_LEN]
    return sanitized.strip("-_.")


def _secret_field_names(config_cls: type[WorkerConfig]) -> frozenset[str]:
    """Return the config fields declared as ``SecretStr``.

    Deriving the credential set from the model keeps it correct as fields are
    added; a hand-maintained list would silently leak the next secret someone
    adds to the worker environment into the pod spec.
    """
    names: set[str] = set()
    for field_name, field in config_cls.model_fields.items():
        annotation = field.annotation
        if any(arg is SecretStr for arg in (annotation, *get_args(annotation))):
            names.add(field_name)
    return frozenset(names)


def _parse_cpu_quantity(value: str) -> int | None:
    raw = value.strip()
    if match := _CPU_MILLI_RE.match(raw):
        return max(1, int(match.group(1)) // 1000)
    try:
        return int(float(raw))
    except ValueError:
        return None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``; mappings merge, everything else replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _prune(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


class KubernetesWorkerConfig(WorkerConfig):
    supervisor_grpc_target: str = (
        f"{env.K8S_SUPERVISOR_SERVICE}.{env.K8S_NAMESPACE}"
        f".svc.{env.K8S_CLUSTER_DOMAIN}:{env.SERVER_GRPC_PORT}"
    )
    """Supervisor gRPC target"""
    flowmesh_url: str = (
        f"http://{env.K8S_SERVER_SERVICE}.{env.K8S_NAMESPACE}"
        f".svc.{env.K8S_CLUSTER_DOMAIN}:{env.SERVER_APP_PORT}"
    )
    """FlowMesh HTTP base URL workers use to fetch and upload artifacts.

    Defaults to the in-cluster Service rather than the externally advertised
    ``FLOWMESH_BASE_URL``, which need not resolve from inside the cluster."""
    namespace: str = env.K8S_WORKER_NAMESPACE
    """Namespace worker pods are created in"""
    pod_name: str | None = None
    """Optional explicit pod name"""
    worker_type: WorkerType = WorkerType.CPU
    """Type of worker (cpu or gpu)"""
    gpu_count: int = 1
    """Number of GPUs requested for GPU workers"""
    gpu_resource_name: str = env.K8S_GPU_RESOURCE_NAME
    """Extended resource name used to request GPUs"""
    gpu_arch: GpuArch | None = None
    """GPU architecture selecting the worker image variant"""
    node_selector: dict[str, str] | None = None
    """Node labels a worker pod must match"""
    tolerations: list[dict[str, Any]] | None = None
    """Tolerations applied to the worker pod"""
    service_account_name: str | None = None
    """Service account the worker pod runs as"""
    image_pull_secrets: list[str] = Field(default_factory=list)
    """Image pull secrets referenced by the worker pod"""
    image_pull_policy: str = "IfNotPresent"
    """Image pull policy for the worker container"""
    cpu_request: str | None = None
    """CPU request for the worker container"""
    cpu_limit: str | None = None
    """CPU limit for the worker container"""
    memory_request: str | None = None
    """Memory request for the worker container"""
    memory_limit: str | None = None
    """Memory limit for the worker container"""
    shm_size: str | None = None
    """Size of the ``/dev/shm`` in-memory volume"""
    results_pvc: str | None = None
    """PersistentVolumeClaim backing the results directory"""
    results_mount_path: str = "/var/lib/flowmesh-results"
    """Path the results volume is mounted at inside the worker container"""
    hf_cache_pvc: str | None = None
    """PersistentVolumeClaim backing the Hugging Face cache"""
    runtime_class_name: str | None = None
    """RuntimeClass for the worker pod"""
    priority_class_name: str | None = None
    """PriorityClass for the worker pod"""
    pod_labels: dict[str, str] | None = None
    """Extra labels applied to the worker pod"""
    pod_annotations: dict[str, str] | None = None
    """Annotations applied to the worker pod"""
    pod_overrides: dict[str, Any] | None = None
    """Overlay merged onto the generated pod manifest.

    Mappings merge recursively; sequences and scalars replace."""
    stop_grace_period_sec: int = 30
    """Termination grace period applied when deleting the worker pod"""
    docker_registry: str = env.FLOWMESH_REGISTRY
    """Registry to pull worker images from"""
    version: str = env.FLOWMESH_VERSION
    """Worker image version tag"""

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        if self.worker_type == WorkerType.GPU and self.gpu_count < 1:
            raise ValueError("Expected at least one GPU for GPU worker.")


class KubernetesWorkerInfo(WorkerInfo):
    pass


class KubernetesWorkerAdapter(WorkerAdapter):
    CONTAINER_HF_CACHE_DIR: str = "/home/appuser/.cache/huggingface"
    SHM_MOUNT_PATH: str = PurePosixPath("/", "dev", "shm").as_posix()
    CONTAINER_NAME: str = "worker"
    WORKER_GID: int = 10001

    def __init__(
        self,
        token: WorkerTokenType,
        name: str,
        pod_name: str,
        config: KubernetesWorkerConfig,
        core_api: client.CoreV1Api,
        node_alias: str,
        owner: PrincipalContext,
    ) -> None:
        super().__init__(token, name, config, owner)

        self.config: KubernetesWorkerConfig
        self.pod_name = pod_name
        self.secret_name = f"{pod_name}-env"
        self.node_alias = node_alias

        self._core = core_api
        self._status: WorkerStatus = WorkerStatus.STOPPED
        self._hardware: dict[str, Any] | WorkerHardware | None = None
        self._is_started = False

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        self._status = status

    def get_info(self) -> KubernetesWorkerInfo:
        hardware = self._hardware
        if isinstance(hardware, dict):
            hardware = WorkerHardware.model_validate(hardware)
            self._hardware = hardware
        return KubernetesWorkerInfo(
            id=self.worker_id,
            name=self.name,
            provider=PROVIDER_NAME,
            status=self.status,
            hardware=hardware,
        )

    async def start(self) -> bool:
        self.set_status(WorkerStatus.STARTING)
        try:
            ok = await asyncio.to_thread(self._start)
            if not ok:
                self.set_status(WorkerStatus.STOPPED)
            return ok
        except Exception:
            self.set_status(WorkerStatus.STOPPED)
            raise

    async def prepare(self) -> None:
        self._hardware = await asyncio.to_thread(self._probe_hardware)

    async def stop(self) -> bool:
        prev_status = self.status
        if prev_status in (WorkerStatus.STOPPING, WorkerStatus.STOPPED):
            return True
        self.set_status(WorkerStatus.STOPPING)
        try:
            ok = await asyncio.to_thread(self._stop)
            if not ok:
                self.set_status(prev_status)
            return ok
        except Exception:
            self.set_status(prev_status)
            raise

    def get_image_name(self) -> str:
        return get_worker_image_name(
            self.config.docker_registry, self.config.version, self._gpu_arch()
        )

    def _gpu_arch(self) -> GpuArch | None:
        if self.config.worker_type != WorkerType.GPU:
            return None
        return self.config.gpu_arch or GpuArch.UNKNOWN

    def _base_environment(self) -> dict[str, str]:
        environment = super()._base_environment()
        environment["RESULTS_DIR"] = self.config.results_mount_path
        return environment

    def _split_environment(self) -> tuple[dict[str, str], dict[str, str]]:
        """Partition the worker environment into inline and secret-held values."""
        secret_keys = {"WORKER_TOKEN", "FLOWMESH_API_KEY"} | {
            name.upper() for name in _secret_field_names(type(self.config))
        }
        inline: dict[str, str] = {}
        secret: dict[str, str] = {}
        for key, value in self._base_environment().items():
            if key in secret_keys:
                secret[key] = value
            else:
                inline[key] = value
        return inline, secret

    def _labels(self) -> dict[str, str]:
        labels = {
            MANAGED_LABEL: "true",
            NODE_ALIAS_LABEL: sanitize_label_value(self.node_alias),
            WORKER_NAME_LABEL: sanitize_label_value(self.name),
        }
        if self.config.pod_labels:
            labels.update(self.config.pod_labels)
        return labels

    def _resources(self) -> dict[str, Any]:
        config = self.config
        requests = _prune({"cpu": config.cpu_request, "memory": config.memory_request})
        limits = _prune({"cpu": config.cpu_limit, "memory": config.memory_limit})
        if config.worker_type == WorkerType.GPU:
            limits[config.gpu_resource_name] = str(config.gpu_count)
        return _prune({"requests": requests or None, "limits": limits or None})

    def _volumes(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        config = self.config
        volumes: list[dict[str, Any]] = []
        mounts: list[dict[str, Any]] = []

        if config.results_pvc:
            volumes.append(
                {
                    "name": "results",
                    "persistentVolumeClaim": {"claimName": config.results_pvc},
                }
            )
        else:
            volumes.append({"name": "results", "emptyDir": {}})
        mounts.append({"name": "results", "mountPath": config.results_mount_path})

        if config.hf_cache_pvc:
            volumes.append(
                {
                    "name": "hf-cache",
                    "persistentVolumeClaim": {"claimName": config.hf_cache_pvc},
                }
            )
            mounts.append(
                {"name": "hf-cache", "mountPath": self.CONTAINER_HF_CACHE_DIR}
            )

        if config.shm_size:
            volumes.append(
                {
                    "name": "dshm",
                    "emptyDir": {"medium": "Memory", "sizeLimit": config.shm_size},
                }
            )
            mounts.append({"name": "dshm", "mountPath": self.SHM_MOUNT_PATH})

        return volumes, mounts

    def build_pod_manifest(self) -> dict[str, Any]:
        config = self.config
        inline_env, _ = self._split_environment()
        volumes, mounts = self._volumes()

        container = _prune(
            {
                "name": self.CONTAINER_NAME,
                "image": self.get_image_name(),
                "imagePullPolicy": config.image_pull_policy,
                "env": [{"name": k, "value": v} for k, v in sorted(inline_env.items())],
                "envFrom": [{"secretRef": {"name": self.secret_name}}],
                "resources": self._resources() or None,
                "volumeMounts": mounts or None,
            }
        )

        spec = _prune(
            {
                "restartPolicy": "Always",
                "terminationGracePeriodSeconds": config.stop_grace_period_sec,
                "containers": [container],
                "volumes": volumes or None,
                "nodeSelector": config.node_selector or None,
                "tolerations": config.tolerations or None,
                "serviceAccountName": config.service_account_name,
                "runtimeClassName": config.runtime_class_name,
                "priorityClassName": config.priority_class_name,
                "securityContext": {"fsGroup": self.WORKER_GID},
                "imagePullSecrets": (
                    [{"name": name} for name in config.image_pull_secrets] or None
                ),
            }
        )

        manifest: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": _prune(
                {
                    "name": self.pod_name,
                    "namespace": config.namespace,
                    "labels": self._labels(),
                    "annotations": config.pod_annotations or None,
                }
            ),
            "spec": spec,
        }
        if config.pod_overrides:
            manifest = _deep_merge(manifest, config.pod_overrides)
        return manifest

    def build_secret_manifest(self) -> dict[str, Any]:
        _, secret_env = self._split_environment()
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "metadata": {
                "name": self.secret_name,
                "namespace": self.config.namespace,
                "labels": self._labels(),
            },
            "stringData": secret_env,
        }

    def _start(self) -> bool:
        existing = self._read_pod()
        if existing is not None:
            phase = self._pod_phase(existing)
            if phase in ("Running", "Pending"):
                self._is_started = True
                logger.warning("Pod %s is already running.", self.pod_name)
                return True
            if not self._delete_pod(grace_period_seconds=0):
                return False
            if not self._await_pod_deletion():
                return False

        if not self._apply_secret():
            return False

        try:
            self._core.create_namespaced_pod(
                namespace=self.config.namespace, body=self.build_pod_manifest()
            )
        except ApiException as exc:
            logger.error("Failed to create pod %s: %s", self.pod_name, repr(exc))
            self._delete_secret()
            return False

        self._is_started = True
        return True

    def _stop(self) -> bool:
        deleted = self._delete_pod()
        self._delete_secret()
        if deleted:
            self._is_started = False
        return deleted

    def _read_pod(self) -> Any | None:
        try:
            return self._core.read_namespaced_pod(
                name=self.pod_name, namespace=self.config.namespace
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            logger.warning("Failed to read pod %s: %s", self.pod_name, repr(exc))
            return None

    @staticmethod
    def _pod_phase(pod: Any) -> str | None:
        status = pod.status if hasattr(pod, "status") else None
        if status is None:
            return None
        return status.phase if hasattr(status, "phase") else None

    def _apply_secret(self) -> bool:
        body = self.build_secret_manifest()
        try:
            self._core.create_namespaced_secret(
                namespace=self.config.namespace, body=body
            )
            return True
        except ApiException as exc:
            if exc.status != 409:
                logger.error(
                    "Failed to create secret %s: %s", self.secret_name, repr(exc)
                )
                return False
        try:
            self._core.replace_namespaced_secret(
                name=self.secret_name, namespace=self.config.namespace, body=body
            )
            return True
        except ApiException as exc:
            logger.error("Failed to update secret %s: %s", self.secret_name, repr(exc))
            return False

    def _await_pod_deletion(self) -> bool:
        """Block until the pod name is free again.

        Deletion is accepted asynchronously and the object outlives the call
        while it terminates, so reusing the name immediately would collide.
        """
        deadline = time.monotonic() + _DELETE_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._read_pod() is None:
                return True
            time.sleep(_DELETE_POLL_SEC)
        logger.error("Pod %s was still terminating after deletion", self.pod_name)
        return False

    def _delete_pod(self, grace_period_seconds: int | None = None) -> bool:
        if grace_period_seconds is None:
            grace_period_seconds = self.config.stop_grace_period_sec
        try:
            self._core.delete_namespaced_pod(
                name=self.pod_name,
                namespace=self.config.namespace,
                grace_period_seconds=grace_period_seconds,
            )
            return True
        except ApiException as exc:
            if exc.status == 404:
                return True
            logger.error("Failed to delete pod %s: %s", self.pod_name, repr(exc))
            return False

    def _delete_secret(self) -> bool:
        try:
            self._core.delete_namespaced_secret(
                name=self.secret_name, namespace=self.config.namespace
            )
            return True
        except ApiException as exc:
            if exc.status == 404:
                return True
            logger.warning(
                "Failed to delete secret %s: %s", self.secret_name, repr(exc)
            )
            return False

    def _probe_hardware(self) -> dict[str, Any] | None:
        selector = ",".join(
            f"{key}={value}" for key, value in (self.config.node_selector or {}).items()
        )
        try:
            nodes = self._core.list_node(label_selector=selector or None)
        except ApiException as exc:
            if exc.status == 403:
                logger.debug(
                    "No permission to list nodes; skipping hardware probe for %s",
                    self.name,
                )
            else:
                logger.warning(
                    "Failed to list nodes for worker %s: %s", self.name, repr(exc)
                )
            return None

        items = nodes.items if hasattr(nodes, "items") else []
        if not items:
            return None
        return self._hardware_from_node(items[0]).model_dump()

    def _hardware_from_node(self, node: Any) -> WorkerHardware:
        status = node.status if hasattr(node, "status") else None
        allocatable = dict(getattr(status, "allocatable", None) or {})
        metadata = node.metadata if hasattr(node, "metadata") else None
        node_labels = dict(getattr(metadata, "labels", None) or {})

        cpu = CPUInfo(logical_cores=_parse_cpu_quantity(allocatable.get("cpu", "")))
        memory = MemoryInfo(
            total_bytes=parse_mem_to_bytes(allocatable.get("memory", ""))
        )

        gpu = GpuPlatformInfo()
        if self.config.worker_type == WorkerType.GPU:
            product = node_labels.get(_GPU_PRODUCT_LABEL, "")
            arch = self.config.gpu_arch or GpuArch.from_name(product)
            gpu = GpuPlatformInfo(
                gpu_arch=arch.value,
                devices=[
                    GpuInfo(index=index, name=product or None)
                    for index in range(self.config.gpu_count)
                ],
            )

        return WorkerHardware(cpu=cpu, memory=memory, gpu=gpu)


class KubernetesWorkerFactory(WorkerFactory):
    def __init__(self, system_principal: PrincipalContext) -> None:
        """Resolve cluster access up front.

        Construction is what decides whether this node reports the provider as
        available, so an unreachable cluster has to fail here rather than at
        first use — otherwise the node would advertise a provider it cannot
        serve and reject the create only once a worker was requested.
        """
        super().__init__(system_principal)
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._api_client = client.ApiClient()
        self._core = client.CoreV1Api(self._api_client)
        self._worker_id_registry: Counter[str] = Counter()
        self._reaped = False

    def create_worker(
        self, token: WorkerTokenType, config: KubernetesWorkerConfig
    ) -> KubernetesWorkerAdapter:
        core = self._core
        self._reap_orphans(core, config.namespace)

        name = config.worker_alias or self._next_worker_name(config.worker_type)
        pod_name = config.pod_name or sanitize_object_name(f"{env.NODE_ALIAS}-{name}")
        return KubernetesWorkerAdapter(
            token=token,
            name=name,
            pod_name=pod_name,
            config=config,
            core_api=core,
            node_alias=env.NODE_ALIAS,
            owner=self.system_principal,
        )

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        if not isinstance(worker, KubernetesWorkerAdapter):
            raise ValueError("Invalid worker type")

    def cleanup(self) -> None:
        self._api_client.close()

    def _next_worker_name(self, worker_type: WorkerType) -> str:
        match worker_type:
            case WorkerType.CPU:
                prefix = "flowmesh_server_worker_cpu_"
            case WorkerType.GPU:
                prefix = "flowmesh_server_worker_gpu_"
            case _:
                raise ValueError(f"Unsupported worker type: {worker_type}")
        self._worker_id_registry[prefix] += 1
        return f"{prefix}{self._worker_id_registry[prefix]}"

    def _reap_orphans(self, core: client.CoreV1Api, namespace: str) -> None:
        """Delete worker pods left behind by a previous supervisor incarnation.

        Their tokens died with that supervisor's in-memory registry, so they can
        never register again; a clean shutdown drains its own workers, so
        anything matching here is a zombie from an unclean exit.
        """
        if self._reaped:
            return
        self._reaped = True

        selector = (
            f"{MANAGED_LABEL}=true,"
            f"{NODE_ALIAS_LABEL}={sanitize_label_value(env.NODE_ALIAS)}"
        )
        try:
            pods = core.list_namespaced_pod(
                namespace=namespace, label_selector=selector
            )
        except ApiException as exc:
            logger.warning("Failed to list orphaned worker pods: %s", repr(exc))
            return

        for pod in pods.items if hasattr(pods, "items") else []:
            pod_name = pod.metadata.name
            logger.info("Reaping orphaned worker pod %s", pod_name)
            try:
                core.delete_namespaced_pod(name=pod_name, namespace=namespace)
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning(
                        "Failed to delete orphaned pod %s: %s", pod_name, repr(exc)
                    )
        try:
            core.delete_collection_namespaced_secret(
                namespace=namespace, label_selector=selector
            )
        except ApiException as exc:
            logger.warning("Failed to delete orphaned worker secrets: %s", repr(exc))


def get_provider_spec(system_principal: PrincipalContext) -> ProviderSpec:
    return ProviderSpec(
        name=PROVIDER_NAME,
        config_cls=KubernetesWorkerConfig,
        adapter_cls=KubernetesWorkerAdapter,
        factory=KubernetesWorkerFactory(system_principal),
    )
