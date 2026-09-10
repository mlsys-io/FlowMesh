"""The `external` worker provider: workers this supervisor does not launch,
admitted by a shared secret (`EXTERNAL_WORKER_TOKEN` / `_FILE`) instead of a
runtime-minted token.

The secret is a bearer credential shared by every external worker: it
authenticates "a worker enrolled by whoever holds the secret", not an
individual machine, cannot be revoked per worker, and admits nobody unless set.
"""

import hmac
from hashlib import sha256

from ... import env
from ...hooks import PrincipalContext
from ..schemas import WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
)

_PROVIDER_NAME = "external"

# A name may itself contain dots, so the split is always on the last one.
_SEP = "."


def mint_external_token(secret: str, name: str) -> WorkerTokenType:
    """Build the token an external worker named `name` must present.

    Deterministic: the same (secret, name) yields the same token, so a worker
    or supervisor restart re-derives the identity instead of losing it.
    """
    digest = hmac.new(secret.encode(), name.encode(), sha256).hexdigest()
    return WorkerTokenType(f"{name}{_SEP}{digest}")


def verify_external_token(token: str, secret: str | None = None) -> str | None:
    """Return the worker name a token proves, or None if it proves nothing.

    Never raises: every rejection path returns None so a caller can treat "not
    an external token" and "a forged one" identically and fall through to the
    normal registry lookup.
    """
    configured = env.EXTERNAL_WORKER_TOKEN if secret is None else secret
    if not configured:
        return None
    name, sep, digest = token.rpartition(_SEP)
    if not sep or not name or not digest:
        return None
    expected = hmac.new(configured.encode(), name.encode(), sha256).hexdigest()
    # Constant-time compare so a forged digest can't be recovered by timing.
    if hmac.compare_digest(expected, digest):
        return name
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
        name: str,
        config: ExternalWorkerConfig,
        owner: PrincipalContext,
    ) -> None:
        super().__init__(token, name, config, owner)
        # An external worker is already running when it presents its token.
        self._status: WorkerStatus = WorkerStatus.RUNNING

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        self._status = status

    def get_info(self) -> WorkerInfo:
        return WorkerInfo(
            id=self.worker_id,
            name=self.name,
            provider=_PROVIDER_NAME,
            status=self._status,
            # Externally managed workers' hardware is not known to the supervisor.
            hardware=None,
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
    def create_worker(
        self, token: WorkerTokenType, config: ExternalWorkerConfig, name: str = ""
    ) -> ExternalWorkerAdapter:
        resolved = name or (verify_external_token(token) or "")
        if not resolved:
            raise ValueError(
                "external worker token does not carry a verifiable name; "
                "it must be minted with mint_external_token()"
            )
        return ExternalWorkerAdapter(token, resolved, config, self.system_principal)

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        # Nothing to release: no container, instance or reservation. The caller
        # removes the registry entry.
        return None


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
