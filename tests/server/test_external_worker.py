"""Tests for the `external` worker provider and its shared-secret admission.

The property under test is the one the feature exists for: a token derived from
CONFIGURATION verifies after the supervisor has forgotten everything, whereas a
runtime-minted `uuid4()` token cannot.
"""

import logging
from threading import Lock
from typing import Any, cast

import grpc
import pytest

from server.clients.redis import (
    WORKERS_SET_KEY,
    SyncRedisClient,
    worker_key,
)
from server.supervisor.adapters.external import (
    ExternalWorkerAdapter,
    ExternalWorkerConfig,
    ExternalWorkerFactory,
    mint_external_token,
    verify_external_token,
)
from server.supervisor.manager import WorkerManager
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.task_listener import TaskListener
from shared.grpc.supervisor.v1 import supervisor_pb2

SECRET = "s3cret-shared-across-the-fleet"


class TestTokenVerification:
    def test_minted_token_verifies_and_yields_the_name(self) -> None:
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, SECRET) == "fm-worker-0"

    def test_token_is_deterministic(self) -> None:
        """The whole point: same inputs, same token, forever."""
        assert mint_external_token(SECRET, "w") == mint_external_token(SECRET, "w")

    def test_wrong_secret_is_rejected(self) -> None:
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, "not-the-secret") is None

    def test_tampered_name_is_rejected(self) -> None:
        """A caller cannot rename themselves into another worker's identity."""
        token = mint_external_token(SECRET, "fm-worker-0")
        _, _, digest = token.rpartition(".")
        assert verify_external_token(f"fm-worker-99.{digest}", SECRET) is None

    def test_names_containing_dots_round_trip(self) -> None:
        """The split is on the LAST dot, so a dotted name is not truncated."""
        name = "site.home.worker-3"
        assert verify_external_token(mint_external_token(SECRET, name), SECRET) == name

    @pytest.mark.parametrize("bad", ["", ".", "nodot", "name.", ".digest"])
    def test_malformed_tokens_return_none_rather_than_raise(self, bad: str) -> None:
        """Callers treat 'not external' and 'forged' identically, so every
        rejection path must return None instead of raising."""
        assert verify_external_token(bad, SECRET) is None

    def test_feature_is_off_when_no_secret_is_configured(self) -> None:
        """With no secret, a well-formed token from ANY secret admits nobody.

        This is what makes the change inert for existing deployments.
        """
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, "") is None

    def test_token_survives_a_supervisor_restart(self) -> None:
        """The regression this feature exists to prevent.

        A runtime-minted token is only ever known to one process: a fresh
        registry (a restarted supervisor) cannot recognise it, which is how a
        healthy worker becomes permanently UNAUTHENTICATED. A config-derived
        token verifies against the secret alone, so it still proves identity
        after the process that first saw it is gone.
        """
        before = WorkerRegistry()
        runtime_token = before.new_token()
        token = mint_external_token(SECRET, "fm-worker-0")

        after_restart = WorkerRegistry()  # everything the old process knew is gone

        assert after_restart.try_get(runtime_token) is None
        assert verify_external_token(token, SECRET) == "fm-worker-0"


class TestExternalAdapter:
    def _adapter(self, name: str = "fm-worker-0") -> ExternalWorkerAdapter:
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        return factory.create_worker(
            mint_external_token(SECRET, name), ExternalWorkerConfig(), name=name
        )

    def test_starts_in_running_because_it_is_already_running(self) -> None:
        assert self._adapter().status is WorkerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_and_stop_are_successful_no_ops(self) -> None:
        """The supervisor neither launches nor kills an external worker, and
        must not report failure for work it was never supposed to do."""
        adapter = self._adapter()
        assert await adapter.start() is True
        assert await adapter.stop() is True

    def test_get_info_reports_the_external_provider_and_no_invented_hardware(
        self,
    ) -> None:
        info = self._adapter().get_info()
        assert info.provider == "external"
        assert info.name == "fm-worker-0"
        #: The supervisor cannot introspect a machine it does not own; a made-up
        #: profile would be fed straight to the scheduler.
        assert info.hardware is None

    def test_create_worker_refuses_a_token_with_no_verifiable_name(self) -> None:
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verifiable name"):
            factory.create_worker("garbage", ExternalWorkerConfig())  # type: ignore[arg-type]

    def test_destroy_does_not_pretend_to_stop_the_process(self) -> None:
        """Forgetting an external worker is accounting, not termination."""
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        adapter = self._adapter()
        factory.destroy_worker(adapter)
        assert adapter.status is WorkerStatus.RUNNING


class TestDockerlessHost:
    """The supervisor must survive a host with no Docker daemon.

    `DockerWorkerFactory.__init__` acquires a client, so before this guard a
    dockerless host raised inside `WorkerManager.__init__` and killed the
    supervisor child -- while the FastAPI parent stayed up answering /healthz,
    which is what made it read Running/Ready with a dead worker plane.
    """

    def test_manager_builds_without_docker_and_keeps_external(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import logging

        from server.supervisor import manager as manager_mod
        from server.supervisor.registry import WorkerRegistry

        def _explode(_principal):
            raise RuntimeError("Error while fetching server API version")

        monkeypatch.setattr(manager_mod, "docker_provider_spec", _explode)
        monkeypatch.setattr(manager_mod, "vastai_provider_spec", _explode)

        mgr = manager_mod.WorkerManager(
            system_principal=None,  # type: ignore[arg-type]
            config_path=str(tmp_path / "absent.yaml"),
            registry=WorkerRegistry(),
            logger=logging.getLogger("test"),
        )

        #: The supervisor is alive and the external provider is usable, which is
        #: exactly the provider a dockerless host needs.
        assert "external" in mgr._providers
        assert "docker" not in mgr._providers

    def test_unavailable_provider_raises_typed_error_and_external_still_works(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import logging

        from server.supervisor import manager as manager_mod
        from server.supervisor.adapters.external import mint_external_token
        from server.supervisor.manager import ProviderUnavailableError, WorkerInitConfig
        from server.supervisor.registry import WorkerRegistry

        def _explode(_principal):
            raise RuntimeError("Error while fetching server API version")

        monkeypatch.setattr(manager_mod, "docker_provider_spec", _explode)
        monkeypatch.setattr(manager_mod, "vastai_provider_spec", _explode)
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)

        mgr = manager_mod.WorkerManager(
            system_principal=None,  # type: ignore[arg-type]
            config_path=str(tmp_path / "absent.yaml"),
            registry=WorkerRegistry(),
            logger=logging.getLogger("test"),
        )
        mgr._is_started = True
        mgr._default_worker_config = {}

        assert mgr.available_providers() == ["external"]

        with pytest.raises(ProviderUnavailableError) as excinfo:
            mgr._create_worker(WorkerInitConfig(provider="docker"))
        assert "docker" in str(excinfo.value)
        assert "external" in str(excinfo.value)

        # The external provider must still work on the same dockerless host.
        token = mint_external_token(SECRET, "fm-worker-0")
        worker = mgr._create_worker(
            WorkerInitConfig(
                provider="external", worker_token=token, init_on_start=False
            )
        )
        assert worker.name == "fm-worker-0"


class _Aborted(Exception):
    """Stand-in for what grpc's ServicerContext.abort raises."""

    def __init__(self, code: grpc.StatusCode) -> None:
        self.code = code


class _FakeContext:
    """Minimal async ServicerContext carrying one x-worker-token."""

    def __init__(self, token: str | None) -> None:
        self._token = token

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return () if self._token is None else (("x-worker-token", self._token),)

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise _Aborted(code)


class _FakeRedis:
    """Records the writes RegisterWorker performs."""

    def __init__(self) -> None:
        self._seq = 0
        self.hashes: dict[str, dict[str, Any]] = {}
        self.worker_ids: set[str] = set()

    def incr(self, key: str) -> int:
        self._seq += 1
        return self._seq

    def sadd(self, key: str, *members: str) -> None:
        if key == WORKERS_SET_KEY:
            self.worker_ids.update(members)

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self.hashes[key] = dict(mapping)


class _FakeTaskListener:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add_worker(self, worker_id: str) -> None:
        self.added.append(worker_id)


def _build_servicer(
    redis: _FakeRedis | None = None,
    node_alias: str = "node-a",
    node_id: str = "nde-1",
) -> tuple[SupervisorServicer, _FakeRedis]:
    redis = redis or _FakeRedis()
    registry = WorkerRegistry()
    manager = WorkerManager(
        cast(Any, None),
        "/nonexistent-worker-config.yaml",
        registry,
        logging.getLogger("test.wm"),
    )
    manager._is_started = True
    manager._default_worker_config = {}
    servicer = SupervisorServicer.__new__(SupervisorServicer)
    servicer._registry = registry
    servicer._redis = cast(SyncRedisClient, redis)
    servicer._node_id = node_id
    servicer._node_alias = node_alias
    servicer._logger = logging.getLogger("test.external.enroll")
    servicer._lock = Lock()
    servicer._task_listener = cast(TaskListener, _FakeTaskListener())
    servicer._worker_manager = manager
    return servicer, redis


async def _register(servicer: SupervisorServicer, token: str) -> str:
    resp = await servicer.RegisterWorker(
        supervisor_pb2.RegisterRequest(), cast(Any, _FakeContext(token))
    )
    return resp.worker_id


class TestRegisterWorkerExternalEnrollment:
    """RegisterWorker is the create+register point for self-enrolling workers."""

    @pytest.mark.asyncio
    async def test_register_enrolls_external_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        worker_id = await _register(servicer, token)

        assert worker_id in redis.worker_ids
        assert redis.hashes[worker_key(worker_id)]["node_id"] == "nde-1"
        assert redis.hashes[worker_key(worker_id)]["node_alias"] == "node-a"
        assert servicer._registry.get_worker_id(cast(Any, token)) == worker_id
        assert cast(_FakeTaskListener, servicer._task_listener).added == [worker_id]

    @pytest.mark.asyncio
    async def test_admit_worker_returns_info_without_starting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manager must NOT run its start lifecycle on an external worker
        (it is already running); admit_worker returns the worker's info rather
        than tripping _start_worker's STOPPED precondition."""
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, _ = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        info = await servicer._worker_manager.admit_worker(cast(Any, token))

        assert info is not None
        assert info.name == "fm-worker-0"
        assert info.provider == "external"
        assert info.status is WorkerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_admit_worker_returns_none_for_non_external_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, _ = _build_servicer()
        info = await servicer._worker_manager.admit_worker(
            servicer._registry.new_token()
        )
        assert info is None

    @pytest.mark.asyncio
    async def test_register_refuses_name_collision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        # A different adapter already owns the name under a different token.
        squatter = ExternalWorkerFactory(system_principal=None).create_worker(  # type: ignore[arg-type]
            mint_external_token("other-secret", "fm-worker-0"),
            ExternalWorkerConfig(),
            name="fm-worker-0",
        )
        servicer._registry.add(squatter)

        token = mint_external_token(SECRET, "fm-worker-0")
        with pytest.raises(_Aborted) as exc:
            await _register(servicer, token)

        assert exc.value.code is grpc.StatusCode.UNAUTHENTICATED
        assert redis.worker_ids == set()

    @pytest.mark.asyncio
    async def test_register_rejects_when_no_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", "")
        servicer, _ = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")
        with pytest.raises(_Aborted) as exc:
            await _register(servicer, token)
        assert exc.value.code is grpc.StatusCode.UNAUTHENTICATED

    @pytest.mark.asyncio
    async def test_register_rejects_forged_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, _ = _build_servicer()
        good = mint_external_token(SECRET, "fm-worker-0")
        _, _, digest = good.rpartition(".")
        with pytest.raises(_Aborted):
            await _register(servicer, f"fm-worker-99.{digest}")

    @pytest.mark.asyncio
    async def test_streams_reject_before_registration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the revert, auth is pure lookup: an external token that has not
        registered resolves to nothing, so every stream RPC aborts."""
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, _ = _build_servicer()
        ctx = cast(Any, _FakeContext(mint_external_token(SECRET, "fm-worker-0")))
        assert servicer._get_worker_from_context(ctx) is None
        assert servicer._get_worker_id_from_context(ctx) is None

    @pytest.mark.asyncio
    async def test_restart_survival_reenrolls_under_a_new_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        redis = _FakeRedis()
        token = mint_external_token(SECRET, "fm-worker-0")

        before, _ = _build_servicer(redis=redis)
        id1 = await _register(before, token)

        # A supervisor restart: fresh in-process registry, same Redis.
        after, _ = _build_servicer(redis=redis)
        ctx = cast(Any, _FakeContext(token))
        assert after._get_worker_id_from_context(ctx) is None  # forgotten

        id2 = await _register(after, token)
        assert id2 != id1
        assert after._get_worker_id_from_context(ctx) == id2

    @pytest.mark.asyncio
    async def test_docker_token_path_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-registered adapter (docker/vastai) registers via try_get and
        never enrolls."""
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        token = servicer._registry.new_token()
        # Simulate WorkerManager having created + registered the adapter already.
        adapter = ExternalWorkerFactory(system_principal=None).create_worker(  # type: ignore[arg-type]
            token, ExternalWorkerConfig(), name="docker-worker-0"
        )
        servicer._registry.add(adapter)

        worker_id = await _register(servicer, cast(str, token))

        assert worker_id in redis.worker_ids
