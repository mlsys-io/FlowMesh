"""The `external` worker provider: workers this supervisor does not launch,
admitted by a shared secret (`EXTERNAL_WORKER_TOKEN` / `_FILE`) instead of a
runtime-minted token.

The secret is a bearer credential shared by every external worker: it
authenticates "a worker enrolled by whoever holds the secret", not an
individual machine, cannot be revoked per worker, and admits nobody unless set.
"""

import hmac
import logging
from hashlib import sha256

from shared.utils.worker_token import EXTERNAL_ALIAS_SEP, split_external_token

from ... import env
from ...hooks import PrincipalContext
from ..resource_manager import ResourceManager
from ..schemas import WorkerHardware, WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
)

_PROVIDER_NAME = "external"

logger = logging.getLogger("supervisor")


def mint_external_token(secret: str, alias: str) -> WorkerTokenType:
    """Build the token an external worker with alias `alias` must present.

    Deterministic: the same (secret, alias) yields the same token, so a worker
    or supervisor restart re-derives the identity instead of losing it.
    """
    digest = hmac.new(secret.encode(), alias.encode(), sha256).hexdigest()
    return WorkerTokenType(f"{alias}{EXTERNAL_ALIAS_SEP}{digest}")


def verify_external_token(token: str, secret: str | None = None) -> str | None:
    """Return the worker alias a token proves, or None if it proves nothing.

    Never raises: every rejection path returns None so a caller can treat "not
    an external token" and "a forged one" identically and fall through to the
    normal registry lookup.
    """
    configured = env.EXTERNAL_WORKER_TOKEN if secret is None else secret
    if not configured:
        return None
    parts = split_external_token(token)
    if parts is None:
        return None
    alias, digest = parts
    expected = hmac.new(configured.encode(), alias.encode(), sha256).hexdigest()
    # Constant-time compare so a forged digest can't be recovered by timing.
    if hmac.compare_digest(expected, digest):
        return alias
    return None


class ExternalWorkerConfig(WorkerConfig):
    """Config for a worker the supervisor does not launch.

    Inherits `WorkerConfig` so `_base_environment()` still describes a valid
    worker environment for operators generating a unit file or Pod spec, though
    the supervisor applies none of it.
    """


class ExternalWorkerAdapter(WorkerAdapter):
    def __init__(
        self,
        token: WorkerTokenType,
        alias: str,
        config: ExternalWorkerConfig,
        owner: PrincipalContext,
    ) -> None:
        super().__init__(token, alias, config, owner)
        # An external worker is already running when it presents its token.
        self._status: WorkerStatus = WorkerStatus.RUNNING
        self._hardware: WorkerHardware | None = None
        self.claimed_gpu_uuids: frozenset[str] | None = None
        """GPU UUIDs the current holds were claimed for; None before a claim."""
        self.held_gpus: list[int] = []
        """Host GPU indices this worker holds in the supervisor's pool."""

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        self._status = status

    def observe_reported_hardware(self, hardware: WorkerHardware) -> None:
        """Keep the worker's own hardware report.

        The supervisor cannot probe a machine it does not own, so the report is
        the only source; `get_info()` reports no hardware until the worker
        registers.
        """
        self._hardware = hardware

    def reported_gpu_uuids(self) -> frozenset[str]:
        if self._hardware is None:
            return frozenset()
        return frozenset(d.uuid for d in self._hardware.gpu.devices if d.uuid)

    def get_info(self) -> WorkerInfo:
        return WorkerInfo(
            id=self.worker_id,
            alias=self.alias,
            provider=_PROVIDER_NAME,
            status=self._status,
            hardware=self._hardware,
            held_gpus=list(self.held_gpus),
        )

    async def start(self) -> bool:
        """No-op: the worker's lifecycle belongs to its orchestrator.

        Reports success because nothing failed; False would mark a healthy
        worker broken.
        """
        return True

    async def stop(self) -> bool:
        """No-op, deliberately not a kill.

        The supervisor cannot stop a process it did not start; conflating
        forget with terminate would make `destroy_worker` silently drop a
        running worker's tasks.
        """
        return True


class ExternalWorkerFactory(WorkerFactory):
    """Creates external workers and holds the host GPUs they report.

    A worker on this supervisor's host holds its GPUs (matched by UUID, since
    the index a worker reports is local to its container) from registration
    until it unregisters or is destroyed. A crash releases nothing: the process
    is usually restarted onto the same cards, which must not have been handed
    to another worker meanwhile. A host without Docker has no pool, so nothing
    is held.
    """

    def __init__(self, system_principal: PrincipalContext) -> None:
        super().__init__(system_principal)
        self._rm: ResourceManager | None
        try:
            self._rm = ResourceManager.get_instance()
        except Exception:
            self._rm = None

    def create_worker(
        self, token: WorkerTokenType, config: ExternalWorkerConfig, alias: str = ""
    ) -> ExternalWorkerAdapter:
        resolved = alias or (verify_external_token(token) or "")
        if not resolved:
            raise ValueError(
                "external worker token does not carry a verifiable alias; "
                "it must be minted with mint_external_token()"
            )
        return ExternalWorkerAdapter(token, resolved, config, self.system_principal)

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        # Releases the GPU holds only: there is no container or instance, and
        # the caller removes the registry entry.
        if isinstance(worker, ExternalWorkerAdapter):
            self._release(worker)

    def on_worker_registered(self, worker: WorkerAdapter) -> bool:
        if not isinstance(worker, ExternalWorkerAdapter):
            return False
        uuids = worker.reported_gpu_uuids()
        if uuids == worker.claimed_gpu_uuids:
            return False
        changed = self._release(worker)
        worker.claimed_gpu_uuids = uuids
        if not uuids or self._rm is None:
            return changed
        claimed, overlapping = self._rm.claim_gpus_by_uuid(uuids)
        worker.held_gpus = sorted(claimed + overlapping)
        if worker.held_gpus:
            logger.info(
                "External worker %s holds host GPUs %s", worker.alias, worker.held_gpus
            )
        if overlapping:
            logger.warning(
                "External worker %s uses host GPUs %s that another worker already "
                "holds; they are now shared",
                worker.alias,
                overlapping,
            )
        return changed or bool(claimed)

    def on_worker_unregistered(self, worker: WorkerAdapter) -> bool:
        if not isinstance(worker, ExternalWorkerAdapter):
            return False
        changed = self._release(worker)
        worker.claimed_gpu_uuids = None
        return changed

    def _release(self, worker: ExternalWorkerAdapter) -> bool:
        devices, worker.held_gpus = worker.held_gpus, []
        if not devices or self._rm is None:
            return False
        before = self._rm.available_gpu_count()
        self._rm.deallocate_gpus(devices)
        logger.info("External worker %s released host GPUs %s", worker.alias, devices)
        return self._rm.available_gpu_count() != before


def get_provider_spec(system_principal: PrincipalContext) -> ProviderSpec:
    return ProviderSpec(
        name=_PROVIDER_NAME,
        config_cls=ExternalWorkerConfig,
        adapter_cls=ExternalWorkerAdapter,
        factory=ExternalWorkerFactory(system_principal),
    )


__all__ = [
    "ExternalWorkerAdapter",
    "ExternalWorkerConfig",
    "ExternalWorkerFactory",
    "get_provider_spec",
    "mint_external_token",
    "verify_external_token",
]
