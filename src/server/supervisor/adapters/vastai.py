import asyncio
import threading
from collections import Counter
from typing import Any

from pydantic import PrivateAttr, SecretStr
from vastai import VastAI  # type: ignore

from ... import env
from ...hooks import PrincipalContext
from ...utils.helpers import ResourcePool, get_logger
from ..resource_manager import GpuArch
from ..schemas import WorkerHardware, WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
)
from .utils import env_to_secret_str, get_worker_image_name, to_env_str

_PROVIDER_NAME = "vastai"

logger = get_logger()


class VastAIWorkerConfig(WorkerConfig):
    supervisor_grpc_target: str = f"{env.SERVER_HOST}:{env.SERVER_GRPC_PORT}"
    """Supervisor gRPC target"""
    instance_id: int | None = None
    """Existing VastAI instance ID to use (if any)"""
    specs: dict[str, str] | None = None
    """VastAI instance specs to query offers
    
    See https://docs.vast.ai/api-reference/search/search-offers for available specs.
    """
    no_default: bool = False
    """Whether to ignore the default specs"""
    disk: float = 10.0
    """Disk size in GB"""
    order: str = "score-"
    """Comma-separated list of fields to sort on. Postfix with - to sort descending"""
    label: str | None = None
    """Label to assign to the VastAI instance"""
    search_limit: int = env.VAST_SEARCH_LIMIT
    """Maximum number of offers to retrieve during search"""

    vast_api_key: SecretStr | None = env_to_secret_str("VAST_API_KEY")
    """VastAI API key"""
    docker_registry: str = env.FLOWMESH_REGISTRY
    """Docker registry to pull worker images from"""
    version: str = env.FLOWMESH_VERSION
    """Worker Docker image version tag"""

    _default_specs: dict[str, str] = PrivateAttr(
        default_factory=lambda: {
            "external": "=false",
            "rentable": "=true",
            "verified": "=true",
        }
    )

    @property
    def hardware_specs(self) -> dict[str, Any]:
        """VastAI hardware specs for instance query"""
        specs: dict[str, Any] = self._build_specs()
        specs["disk"] = self.disk
        return specs

    @property
    def query(self) -> str:
        """VastAI instance query string"""
        specs = self._build_specs()
        return " ".join(f"{k}{v}" for k, v in specs.items())

    def _build_specs(self) -> dict[str, str]:
        specs = {} if self.no_default else self._default_specs.copy()
        if self.specs:
            specs.update(self.specs)
        if "disk_space" not in specs:
            specs["disk_space"] = f">={self.disk}"
        return specs


class VastAIWorkerInfo(WorkerInfo):
    pass


class VastAIWorkerAdapter(WorkerAdapter):
    CONTAINER_RESULTS_DIR: str = "/var/lib/flowmesh-results"
    _STOP_TIMEOUT: float = 60.0

    def __init__(
        self,
        token: WorkerTokenType,
        name: str,
        config: VastAIWorkerConfig,
        vastai_client: VastAI,
        instance_pool: ResourcePool[int],
        owner: PrincipalContext,
    ) -> None:
        super().__init__(token, name, config, owner)
        self.config: VastAIWorkerConfig
        self._client = vastai_client
        self._instance_pool = instance_pool
        self._status: WorkerStatus = WorkerStatus.STOPPED
        self._instance_id: int | None = config.instance_id
        self._created_instance = False
        self._hardware: dict[str, Any] | WorkerHardware | None = None
        self._reserved_offer_id: int | None = None

        self._stop_event: threading.Event = threading.Event()

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        match status:
            case WorkerStatus.STARTING:
                self._stop_event.clear()
            case WorkerStatus.STOPPED:
                self._stop_event.set()
        self._status = status

    def get_info(self) -> VastAIWorkerInfo:
        hardware = self._hardware
        if isinstance(hardware, dict):
            hardware = WorkerHardware.model_validate(hardware)
            self._hardware = hardware
        return VastAIWorkerInfo(
            id=self.worker_id,
            name=self.name,
            provider=_PROVIDER_NAME,
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
        if self._hardware is None:
            instance_id = self._instance_id
            if instance_id is None:
                hardware = self.config.hardware_specs
            else:
                instance_info = self._get_instance_info(instance_id)
                if instance_info is None:
                    raise ValueError(
                        f"Unable to find VastAI instance with ID {instance_id}"
                    )
                hardware = instance_info
            self._hardware = hardware

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

    def _base_environment(self) -> dict[str, str]:
        environment = super()._base_environment()
        environment["RESULTS_DIR"] = self.CONTAINER_RESULTS_DIR
        return environment

    def _start(self) -> bool:
        instance_id = self.config.instance_id
        if instance_id is not None:
            # Create from existing instance
            instance_info = self._get_instance_info(instance_id)
            if instance_info is None:
                logger.error(
                    "Unable to find VastAI instance with ID %s for worker %s.",
                    instance_id,
                    self.name,
                )
                return False
            logger.debug(
                "Starting existing VastAI instance %s for worker %s.",
                instance_id,
                self.name,
            )
            err = self._client.start_instance(id=instance_id)
            if err is not None:
                logger.error(
                    "Failed to start VastAI instance %s for worker %s: %s",
                    instance_id,
                    self.name,
                    err,
                )
                return False
            self._instance_id = instance_id
            self._created_instance = False
            self._hardware = instance_info
            return True

        # Try to create new instance
        max_retries = env.VAST_MAX_RETRIES
        for i in range(1, max_retries + 1):
            try:
                return self._create_and_start()
            except Exception as exc:
                logger.warning(
                    "Failed to start VastAI worker (attempt %d): %s", i, repr(exc)
                )
        logger.error("Exceeded maximum VastAI worker start attempts (%d).", max_retries)
        return False

    def _create_and_start(self) -> bool:
        config = self.config
        assert config.instance_id is None
        instance_pool = self._instance_pool
        candidate_instances: dict[int, dict[str, Any]]
        # Search offers
        offers: list[dict[str, Any]] = self._client.search_offers(
            query=config.query,
            limit=config.search_limit,
            storage=config.disk,
            order=config.order,
        )  # type: ignore
        if len(offers) == 0:
            raise ValueError(f"No VastAI offers found matching specs: {config.query}")
        candidate_instances = {offer["id"]: offer for offer in offers}

        for instance_id, instance_info in candidate_instances.items():
            if not instance_pool.reserve(instance_id):
                continue
            logger.debug(
                "Launching VastAI instance %s for worker %s.", instance_id, self.name
            )
            gpu_name = instance_info.get("gpu_name")
            gpu_arch = None if gpu_name is None else GpuArch.from_name(gpu_name)
            env = self._build_env_str()
            try:
                resp = self._client.create_instance(
                    id=instance_id,
                    disk=config.disk,
                    image=get_worker_image_name(
                        config.docker_registry, config.version, gpu_arch
                    ),
                    label=config.label or self.name,
                    env=env,
                    cancel_unavail=True,
                    # NOTE(kaiitunnz): The following come from worker's Dockerfile
                    entrypoint="/usr/bin/tini --",
                    onstart_cmd="/app/worker/entrypoint.sh",
                )  # type: ignore
            except Exception:
                instance_pool.release(instance_id)
                raise
            if not isinstance(resp, dict):
                logger.debug(
                    "Failed to create VastAI instance %s for worker %s: %s",
                    instance_id,
                    self.name,
                    resp,
                )
                instance_pool.release(instance_id)
                continue
            if not resp.get("success"):
                logger.debug(
                    "Failed to create VastAI instance %s for worker %s: %s",
                    instance_id,
                    self.name,
                    resp,
                )
                instance_pool.release(instance_id)
                continue

            new_instance_id = resp.get("new_contract", instance_id)
            hardware = self._get_instance_info(new_instance_id)
            if hardware is None:
                logger.debug(
                    "Failed to retrieve info for VastAI instance %s. "
                    "Falling back to using the offer info",
                    new_instance_id,
                )
                hardware = instance_info
            self._instance_id = new_instance_id
            self._created_instance = True
            self._hardware = hardware
            self._reserved_offer_id = instance_id
            logger.debug(
                "Successfully created VastAI instance %s for worker %s.",
                new_instance_id,
                self.name,
            )
            return True
        raise RuntimeError("Failed to launch any VastAI instance.")

    def _stop(self) -> bool:
        instance_id = self._instance_id
        if instance_id is None:
            return True

        if self._created_instance:
            logger.debug(
                "Destroying VastAI instance %s created for worker %s.",
                instance_id,
                self.name,
            )
            err = self._client.destroy_instance(id=instance_id)
        else:
            logger.debug(
                "Stopping VastAI instance %s for worker %s.",
                instance_id,
                self.name,
            )
            err = self._client.stop_instance(id=instance_id)
        if err is not None:
            logger.error(
                "Failed to stop VastAI instance %s for worker %s: %s",
                instance_id,
                self.name,
                err,
            )
            self._release_reserved_offer()
            return False
        self._stop_event.wait(self._STOP_TIMEOUT)
        self._release_reserved_offer()
        self._instance_id = None
        self._hardware = self.config.hardware_specs
        return True

    def _build_env_str(self) -> str:
        env = self._base_environment()
        parts: list[str] = []
        for key, val in env.items():
            if val is None:
                continue
            parts.append(f"-e {key}={to_env_str(val)}")
        return " ".join(parts)

    def _get_instance_info(self, instance_id: int) -> dict[str, Any] | None:
        try:
            instance = self._client.show_instance(id=instance_id)
        except Exception:
            return None
        if isinstance(instance, dict):
            instance.pop("actual_status", None)
            return instance
        return None

    def _release_reserved_offer(self) -> None:
        offer_id = self._reserved_offer_id
        if offer_id is None:
            return
        self._instance_pool.release(offer_id)
        self._reserved_offer_id = None


class VastAIWorkerFactory(WorkerFactory):
    def __init__(self, system_principal: PrincipalContext) -> None:
        super().__init__(system_principal)
        self._client_cache: dict[str, VastAI] = {}
        self._worker_id_registry: Counter[str] = Counter()
        self._instance_pool = ResourcePool[int]()

    def create_worker(
        self, token: WorkerTokenType, config: VastAIWorkerConfig
    ) -> VastAIWorkerAdapter:
        api_key = (
            config.vast_api_key.get_secret_value() if config.vast_api_key else None
        )
        if not api_key:
            raise ValueError("VastAI API key is required to create a VastAI worker.")

        client = self._client_cache.get(api_key)
        if client is None:
            client = VastAI(api_key=api_key, raw=True, quiet=True)
            self._client_cache[api_key] = client
        return VastAIWorkerAdapter(
            token=token,
            name=self._resolve_worker_name(config),
            config=config,
            vastai_client=client,
            instance_pool=self._instance_pool,
            owner=self.system_principal,
        )

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        if not isinstance(worker, VastAIWorkerAdapter):
            raise ValueError("Invalid worker type")
        # VastAI instances are torn down during stopping; nothing to release here.
        return

    def _resolve_worker_name(self, config: VastAIWorkerConfig) -> str:
        if config.label is not None:
            return config.label
        if config.worker_alias is not None:
            return config.worker_alias
        return self._get_next_worker_name()

    def _get_next_worker_name(self) -> str:
        prefix = "flowmesh_vastai_worker_"
        next_id = self._worker_id_registry[prefix]
        self._worker_id_registry[prefix] += 1
        return f"{prefix}{next_id}"


def get_provider_spec(system_principal: PrincipalContext) -> ProviderSpec:
    return ProviderSpec(
        name=_PROVIDER_NAME,
        config_cls=VastAIWorkerConfig,
        adapter_cls=VastAIWorkerAdapter,
        factory=VastAIWorkerFactory(system_principal),
    )
