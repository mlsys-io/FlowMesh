"""The `external` worker provider: workers this supervisor does not own.

WHY THIS EXISTS
---------------
Every other provider *creates* the worker it admits — `docker` starts a
container, `vastai` rents an instance — and mints that worker a token with
`WorkerRegistry.new_token()`, which is `uuid.uuid4().hex` held in a plain dict
in the supervisor process. Two consequences follow from that, and both are
load-bearing bugs for anyone running workers under an external lifecycle
manager (Kubernetes, systemd, nomad, a human with a terminal):

1. **A supervisor restart orphans every live worker, permanently.** The dict is
   the only record of the token. When the process restarts the dict is empty,
   so a worker that is running perfectly and heartbeating gets
   `UNAUTHENTICATED` on every RPC forever. It does not recover, and — because
   `_run_event_stream` / `_run_task_stream` catch `grpc.RpcError`, sleep 3s and
   retry unconditionally — it does not exit either. It becomes a zombie that
   looks alive to its own orchestrator while the supervisor will never dispatch
   to it again. Measured in the field: two supervisor restarts left 6 ghost
   registry rows against 2 live workers, and one deployment logged 33,147
   restarts against this shape.

2. **A worker cannot be admitted at all without the supervisor spawning it.**
   The only way to obtain a token is `POST /api/v1/stack/workers`, which builds
   a provider adapter — and on a host with no Docker daemon the `docker`
   provider cannot even be constructed (see `manager.py`), so the endpoint is
   unreachable on exactly the hosts where an external worker makes most sense.

HOW THIS PROVIDER FIXES BOTH
----------------------------
Admission is by a **shared secret** that lives in configuration
(`EXTERNAL_WORKER_TOKEN`, or `EXTERNAL_WORKER_TOKEN_FILE` for a mounted k8s
Secret) rather than by a value generated at runtime. A worker presents

    <name>.<hmac_sha256(secret, name)>

so the supervisor can verify it **statelessly**: split on the last dot,
recompute the HMAC over the name, compare in constant time. No dict lookup, no
stored state, nothing to lose across a restart. The same worker presenting the
same token after a supervisor restart is re-admitted and re-attached to its
streams, which is precisely the failure above, gone.

The name is carried *inside* the token rather than alongside it because
`StreamTasks` and `PushEvents` send only `x-worker-token` metadata — there is
no register payload on those RPCs to read a name from, and they are exactly the
calls that must survive a restart.

WHAT THIS ADAPTER DELIBERATELY DOES NOT DO
------------------------------------------
It owns no container and never imports `docker`. `start()` and `stop()` are
no-ops that report success, because the worker's lifecycle belongs to whoever
runs it; the supervisor's job here is to admit, route and account for it. That
is also why `destroy_worker` does not try to kill anything: forgetting an
external worker must never be confused with terminating it.

SECURITY NOTE, STATED PLAINLY
-----------------------------
This is a bearer secret shared by every external worker on a supervisor. It
authenticates "a worker enrolled by whoever holds the secret", not an
individual machine, and it cannot be revoked per worker — rotating it re-admits
every worker. That is a deliberate trade for a token that survives restarts,
and it is why the feature is **off unless the secret is configured**: with no
secret set, `verify_external_token` returns None for every input and this
provider admits nobody.
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

#: Separator between the worker name and its HMAC. A name may itself contain
#: dots, so the split is on the LAST occurrence, never the first.
_SEP = "."


def mint_external_token(secret: str, name: str) -> WorkerTokenType:
    """Build the token an external worker named `name` must present.

    Deterministic on purpose: the same (secret, name) pair yields the same
    token forever, which is the whole point — a worker restarted by its
    orchestrator, or a supervisor restarted under running workers, re-derives
    the identity instead of losing it.
    """
    digest = hmac.new(secret.encode(), name.encode(), sha256).hexdigest()
    return WorkerTokenType(f"{name}{_SEP}{digest}")


def verify_external_token(token: str, secret: str | None = None) -> str | None:
    """Return the worker name a token proves, or None if it proves nothing.

    Returns None — never raises — for every rejection path, so a caller can
    treat "not an external token" and "a forged external token" identically and
    fall through to the normal registry lookup.
    """
    configured = env.EXTERNAL_WORKER_TOKEN if secret is None else secret
    if not configured:
        #: The feature is off. Admit nobody, regardless of what was presented.
        return None
    name, sep, digest = token.rpartition(_SEP)
    if not sep or not name or not digest:
        return None
    expected = hmac.new(configured.encode(), name.encode(), sha256).hexdigest()
    #: compare_digest, not ==, so a forged token cannot be recovered one byte
    #: at a time by timing the comparison.
    if not hmac.compare_digest(expected, digest):
        return None
    return name


class ExternalWorkerConfig(WorkerConfig):
    """Config for a worker the supervisor does not launch.

    Inherits every field of `WorkerConfig` so that `_base_environment()` still
    describes a valid worker environment — useful for operators generating a
    unit file or a Pod spec — but nothing here is applied by the supervisor,
    because it never starts the process.
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
        #: An external worker is, by definition, already running when it
        #: presents its token — it registered itself. Anything else would be a
        #: lie the dispatcher acts on.
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
            #: Hardware is reported by the worker itself in its register
            #: payload; the supervisor cannot introspect a machine it does not
            #: own, and inventing a profile here would feed the scheduler a
            #: number nobody measured.
            hardware=None,
        )

    async def start(self) -> bool:
        """No-op: the worker's lifecycle belongs to its orchestrator.

        Reports success because from the supervisor's side there is nothing to
        do and nothing failed. Returning False would mark a healthy worker as
        broken.
        """
        return True

    async def stop(self) -> bool:
        """No-op, and deliberately NOT a kill.

        The supervisor cannot stop a process it did not start. Forgetting an
        external worker is an accounting action; conflating it with termination
        would make `destroy_worker` silently lose a still-running worker's
        tasks.
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
        #: Nothing to release: no container, no rented instance, no GPU
        #: reservation. The registry entry is removed by the caller.
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
