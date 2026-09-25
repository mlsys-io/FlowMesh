import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..hooks import PrincipalContext
from .adapters.base import ProviderSpec, WorkerAdapter, WorkerTokenType
from .adapters.docker import get_provider_spec as docker_provider_spec
from .adapters.external import get_provider_spec as external_provider_spec
from .adapters.external import verify_external_token
from .adapters.vastai import get_provider_spec as vastai_provider_spec
from .registry import WorkerRegistry
from .schemas import WorkerInfo, WorkerStatus

_MAX_PARALLELISM: int = 16


class ManagerNotStartedError(RuntimeError):
    """Raised when an operation arrives before the manager has started."""

    def __init__(self, message: str = "WorkerManager not started") -> None:
        super().__init__(message)


class ProviderUnavailableError(ValueError):
    """Raised when a create request names a provider this node does not have."""


class WorkerInitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="docker", description="Worker provider")
    init_on_start: bool = Field(
        default=True,
        description="Whether to start the worker immediately",
    )
    worker_token: WorkerTokenType | None = Field(
        default=None, description="Optional worker token (overrides registry token)"
    )
    worker_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific worker config"
    )

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in {"provider", "init_on_start", "worker_token", "worker_config"}
        }


class ServerWorkerConfig(BaseModel):
    default_worker_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration applied to all workers",
    )
    workers: list[WorkerInitConfig] = Field(
        default_factory=list,
        description="List of worker configurations",
    )


class WorkerManager:
    def __init__(
        self,
        system_principal: PrincipalContext,
        config_path: str,
        registry: WorkerRegistry,
        logger: logging.Logger,
        capacity_change_callback: Callable[[], None] | None = None,
    ) -> None:
        self.config_path = config_path
        self.logger = logger

        self._registry = registry
        self._default_worker_config: dict[str, Any] | None = None
        self._is_started: bool = False
        self._capacity_change_callback = capacity_change_callback
        # External provider is always available.
        specs = [external_provider_spec(system_principal)]
        try:
            specs.append(docker_provider_spec(system_principal))
        except Exception as exc:
            logger.warning(
                "Docker worker provider unavailable, continuing without it: %s", exc
            )
        try:
            specs.append(vastai_provider_spec(system_principal))
        except Exception as exc:
            logger.warning(
                "Vast.ai worker provider unavailable, continuing without it: %s", exc
            )
        self._providers: dict[str, ProviderSpec] = {spec.name: spec for spec in specs}

    @property
    def is_started(self) -> bool:
        return self._is_started

    async def start(self) -> None:
        if self.is_started:
            self.logger.warning("WorkerManager is already started.")
            return

        self._is_started = True
        self._default_worker_config = {}

        if not os.path.isfile(self.config_path):
            self.logger.warning(
                (
                    "Worker config file '%s' does not exist. "
                    "Skipping worker initialization."
                ),
                self.config_path,
            )
            return

        # Load worker configs from the config file
        with open(self.config_path, encoding="utf-8") as f:
            raw = f.read()
        config_data = yaml.safe_load(raw) if raw.strip() else None
        if config_data is None:
            self.logger.info(
                "Worker config file '%s' is empty. Skipping worker initialization.",
                self.config_path,
            )
            return

        server_config = ServerWorkerConfig.model_validate(config_data)
        self._default_worker_config = server_config.default_worker_config

        to_start: list[WorkerAdapter] = []
        to_prepare: list[WorkerAdapter] = []
        for init_config in server_config.workers:
            try:
                worker = self._create_worker(init_config)
                worker_info = worker.get_info()
                self.logger.info(
                    "Created worker %s with provider '%s' (status=%s).",
                    worker_info.alias,
                    worker_info.provider,
                    worker_info.status,
                )
                if init_config.init_on_start:
                    to_start.append(worker)
                else:
                    to_prepare.append(worker)
            except Exception as exc:
                self.logger.error("Failed to register worker: %s", exc)

        if not (to_start or to_prepare):
            return

        max_parallel = min(len(to_start) + len(to_prepare), _MAX_PARALLELISM)
        sema = asyncio.Semaphore(max_parallel or 1)
        coros: list[Awaitable] = []
        if to_start:

            async def start_worker(worker: WorkerAdapter) -> None:
                async with sema:
                    try:
                        await self._start_worker(worker)
                    except Exception as exc:
                        self.logger.error(
                            "Failed to start worker %s: %s", worker.alias, exc
                        )

            coros.extend(start_worker(worker) for worker in to_start)

        if to_prepare:

            async def prepare_worker(worker: WorkerAdapter) -> None:
                async with sema:
                    try:
                        await worker.prepare()
                    except Exception as exc:
                        self.logger.error(
                            "Failed to prepare worker %s: %s", worker.alias, exc
                        )

            coros.extend(prepare_worker(worker) for worker in to_prepare)

        await asyncio.gather(*coros)
        self._report_capacity_change()

    async def stop(self) -> None:
        if not self.is_started:
            self.logger.warning("WorkerManager is not started.")
            return

        await self._stop_and_destroy_workers(self._registry.all_workers())
        self._report_capacity_change()
        for spec in self._providers.values():
            spec.factory.cleanup()
        self._registry.clear()
        self._default_worker_config = None
        self._is_started = False
        self.logger.info("Worker manager stopped")

    async def create_worker(self, init_config: WorkerInitConfig) -> WorkerInfo:
        if not self.is_started:
            raise ManagerNotStartedError()

        worker = self._create_worker(init_config)
        if init_config.init_on_start:
            # A worker created here must not hold its alias or GPUs against a retry.
            try:
                if not await self._start_worker(worker):
                    raise RuntimeError(f"Failed to start worker '{worker.alias}'")
            except Exception:
                await self._stop_and_destroy_worker(worker)
                self._registry.try_pop(worker.token)
                raise
        self._report_capacity_change()
        return worker.get_info()

    async def admit_worker(self, token: WorkerTokenType) -> WorkerInfo | None:
        if not self.is_started:
            raise ManagerNotStartedError()

        if verify_external_token(token) is None:
            return None
        try:
            # init_on_start=False: an external worker is already running, so the
            # supervisor must not run its start lifecycle on it (that path gates
            # on STOPPED and would reject a worker born RUNNING).
            return await self.create_worker(
                WorkerInitConfig(
                    provider="external", worker_token=token, init_on_start=False
                )
            )
        except ValueError as exc:
            self.logger.warning("Failed to admit worker: %s", exc)
            return None

    def available_providers(self) -> list[str]:
        return sorted(self._providers)

    def list_workers(self) -> list[WorkerInfo]:
        if not self.is_started:
            return []
        return [worker.get_info() for worker in self._registry.all_workers()]

    def get_worker_info(self, alias: str) -> WorkerInfo | None:
        if not self.is_started:
            return None
        worker = self._registry.try_get_by_alias(alias)
        return None if worker is None else worker.get_info()

    async def start_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            raise ValueError(f"Worker '{alias}' does not exist")

        return await self._start_worker(worker)

    async def stop_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            raise ValueError(f"Worker '{alias}' does not exist")
        if worker.status not in (WorkerStatus.STARTING, WorkerStatus.RUNNING):
            raise ValueError(f"Worker '{alias}' is not starting or running")

        return await self._stop_worker(worker)

    async def destroy_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            return False

        success = await self._stop_and_destroy_worker(worker)
        self._registry.try_pop_by_alias(alias)
        self._report_capacity_change()
        return success

    async def destroy_workers(self, aliases: set[str] | None = None) -> None:
        if not self.is_started:
            raise ManagerNotStartedError()

        workers: list[WorkerAdapter]
        if aliases is None:
            workers = self._registry.all_workers()
        else:
            missing = [
                alias for alias in aliases if not self._registry.exists_by_alias(alias)
            ]
            if missing:
                raise ValueError(f"Workers not found: {', '.join(missing)}")
            workers = [self._registry.get_by_alias(alias) for alias in aliases]

        await self._stop_and_destroy_workers(workers)
        if aliases is None:
            self._registry.clear()
        else:
            for alias in aliases:
                self._registry.try_pop_by_alias(alias)
        self._report_capacity_change()

    def _create_worker(self, init_config: WorkerInitConfig) -> WorkerAdapter:
        if not self.is_started:
            raise ManagerNotStartedError()

        token = init_config.worker_token or self._registry.new_token()
        provider = init_config.provider.strip().lower()
        worker_config = (self._default_worker_config or {}) | init_config.worker_config

        spec = self._providers.get(provider)
        if spec is None:
            raise ProviderUnavailableError(
                f"Worker provider '{provider}' is not available on this node; "
                f"available providers: {', '.join(sorted(self._providers))}"
            )
        config = spec.config_cls.model_validate(worker_config)
        worker = spec.factory.create_worker(token, config)

        try:
            self._registry.add(worker)
        except ValueError:
            self._destroy_worker(worker)
            raise ValueError(f"Worker '{worker.alias}' already exists")
        return worker

    async def _start_worker(self, worker: WorkerAdapter) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        if worker.status is not WorkerStatus.STOPPED:
            raise ValueError(f"Worker '{worker.alias}' is already started")

        started = await worker.start()
        if not started:
            self.logger.error("Worker %s failed to start", worker.alias)
            return False
        return True

    def _destroy_worker(self, worker: WorkerAdapter) -> None:
        for spec in self._providers.values():
            if isinstance(worker, spec.adapter_cls):
                spec.factory.destroy_worker(worker)
                return
        raise ValueError(f"Unsupported worker type: {type(worker)}")

    def _report_capacity_change(self) -> None:
        callback = self._capacity_change_callback
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self.logger.debug("Failed to report capacity change: %s", exc)

    async def _stop_and_destroy_workers(self, workers: list[WorkerAdapter]) -> None:
        if not workers:
            return

        max_workers = min(len(workers), _MAX_PARALLELISM)
        sema = asyncio.Semaphore(max_workers or 1)

        async def stop_and_destroy(worker: WorkerAdapter) -> None:
            async with sema:
                await self._stop_and_destroy_worker(worker)

        await asyncio.gather(*(stop_and_destroy(worker) for worker in workers))

    async def _stop_and_destroy_worker(self, worker: WorkerAdapter) -> bool:
        worker_alias = worker.alias
        success = True
        was_running = worker.status in (WorkerStatus.STARTING, WorkerStatus.RUNNING)

        if was_running:
            self.logger.info("Stopping worker %s...", worker_alias)
            try:
                success = await worker.stop()
            except Exception as exc:
                self.logger.error(
                    "Failed to stop worker %s: %s", worker_alias, repr(exc)
                )
                success = False
        else:
            self.logger.info("Destroying worker %s that is not running.", worker_alias)

        try:
            self._destroy_worker(worker)
        except Exception as exc:
            self.logger.error(
                "Failed to destroy worker %s: %s", worker_alias, repr(exc)
            )
            success = False

        if success:
            outcome = "stopped" if was_running else "destroyed"
            self.logger.info("Worker %s %s.", worker_alias, outcome)

        return success

    async def _stop_worker(self, worker: WorkerAdapter) -> bool:
        worker_alias = worker.alias
        if worker.status not in (WorkerStatus.STARTING, WorkerStatus.RUNNING):
            raise ValueError(f"Worker '{worker_alias}' is not starting or running")

        self.logger.info("Stopping worker %s...", worker_alias)
        try:
            success = await worker.stop()
            if success:
                if self._registry.get_worker_id(worker.token) is None:
                    # Ensure unregistered workers are restartable after stopped.
                    worker.set_status(WorkerStatus.STOPPED)
                self.logger.info("Worker %s stopped.", worker_alias)
            else:
                self.logger.error("Failed to stop worker %s", worker_alias)
            return success
        except Exception as exc:
            self.logger.error("Failed to stop worker %s: %s", worker_alias, repr(exc))
            return False
