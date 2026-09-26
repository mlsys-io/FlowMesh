"""Tests for the `external` worker provider and its shared-secret admission.

The property under test is the one the feature exists for: a token derived from
CONFIGURATION verifies after the supervisor has forgotten everything, whereas a
runtime-minted `uuid4()` token cannot.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from threading import Lock
from typing import Any, cast

import grpc
import pytest

from server.clients.redis import (
    WORKERS_SET_KEY,
    SyncRedisClient,
    worker_key,
)
from server.hooks import PrincipalContext
from server.schemas.node import NodeWorkerInfo
from server.supervisor.adapters.external import (
    ExternalWorkerAdapter,
    ExternalWorkerConfig,
    ExternalWorkerFactory,
    mint_external_token,
    verify_external_token,
)
from server.supervisor.manager import (
    ProviderUnavailableError,
    WorkerInitConfig,
    WorkerManager,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.resource_manager import GpuArch, MachineEnv, ResourceManager
from server.supervisor.schemas import WorkerHardware, WorkerStatus
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.relay_service import RelayService
from server.supervisor.services.task_listener import TaskListener
from shared.grpc.supervisor.v1 import supervisor_pb2
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
)
from shared.tasks.worker_message import WorkerHardware as ReportedHardware

SECRET = "s3cret-shared-across-the-fleet"


def _hardware_json(*gpu_uuids: str) -> str:
    """The `hardware_json` a worker sends, as `worker.hw` builds it."""
    return ReportedHardware(
        cpu=CPUInfo(logical_cores=8, model="x86_64"),
        memory=MemoryInfo(total_bytes=64 << 30),
        gpu=GpuPlatformInfo(
            driver_version="550.54",
            cuda_version="12.4",
            devices=[
                GpuInfo(
                    index=i,
                    name="NVIDIA H100",
                    uuid=uuid,
                    memory_total_bytes=80 << 30,
                    memory_free_bytes=80 << 30,
                    gpu_available=True,
                )
                for i, uuid in enumerate(gpu_uuids)
            ],
        ),
        network=NetworkInfo(ip="10.0.0.5", bandwidth_bytes_per_sec=None),
    ).model_dump_json()


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
            mint_external_token(SECRET, name), ExternalWorkerConfig(), alias=name
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
        assert info.alias == "fm-worker-0"
        #: The supervisor cannot introspect a machine it does not own, so there
        #: is no hardware until the worker reports its own.
        assert info.hardware is None
        assert info.held_gpus == []

    def test_get_info_reports_the_hardware_the_worker_reported(self) -> None:
        adapter = self._adapter()
        adapter.observe_reported_hardware(
            WorkerHardware.model_validate_json(_hardware_json())
        )
        info = adapter.get_info()
        assert info.hardware is not None
        assert info.hardware.cpu.logical_cores == 8

    def test_create_worker_refuses_a_token_with_no_verifiable_name(self) -> None:
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verifiable alias"):
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
        from server.supervisor import manager as manager_mod

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
        assert worker.alias == "fm-worker-0"


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


class _FakeRelayService:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add_event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


def _build_servicer(
    redis: _FakeRedis | None = None,
    node_alias: str = "node-a",
    node_id: str = "nde-1",
    resource_manager: ResourceManager | None = None,
    capacity_change_callback: Callable[[], None] | None = None,
) -> tuple[SupervisorServicer, _FakeRedis]:
    redis = redis or _FakeRedis()
    registry = WorkerRegistry()
    manager = WorkerManager(
        cast(Any, None),
        "/nonexistent-worker-config.yaml",
        registry,
        logging.getLogger("test.wm"),
        capacity_change_callback=capacity_change_callback,
    )
    factory = manager._providers["external"].factory
    assert isinstance(factory, ExternalWorkerFactory)
    factory._rm = resource_manager
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
    servicer._relay_service = cast(RelayService, _FakeRelayService())
    return servicer, redis


async def _register(
    servicer: SupervisorServicer,
    token: str,
    alias: str | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    request = supervisor_pb2.RegisterRequest()
    if alias is not None:
        request.meta.update({"alias": alias})
    if meta is not None:
        request.meta.update(meta)
    resp = await servicer.RegisterWorker(request, cast(Any, _FakeContext(token)))
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
        assert info.alias == "fm-worker-0"
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
            alias="fm-worker-0",
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
            token, ExternalWorkerConfig(), alias="docker-worker-0"
        )
        servicer._registry.add(adapter)

        worker_id = await _register(servicer, cast(str, token))

        assert worker_id in redis.worker_ids


class TestRegisterWorkerRecordsReportedHardware:
    """The supervisor's view of an external worker's hardware is the worker's
    own registration report."""

    @pytest.mark.asyncio
    async def test_register_records_reported_hardware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, _ = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        await _register(
            servicer, token, meta={"hardware_json": _hardware_json("GPU-aaa")}
        )

        info = servicer._worker_manager.get_worker_info("fm-worker-0")
        assert info is not None and info.hardware is not None
        assert info.hardware.cpu.logical_cores == 8
        assert [d.uuid for d in info.hardware.gpu.devices] == ["GPU-aaa"]

    @pytest.mark.asyncio
    async def test_unreadable_hardware_does_not_fail_registration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        worker_id = await _register(servicer, token, meta={"hardware_json": "{oops"})

        assert worker_id in redis.worker_ids
        info = servicer._worker_manager.get_worker_info("fm-worker-0")
        assert info is not None and info.hardware is None

    @pytest.mark.asyncio
    async def test_reenrollment_after_restart_restores_hardware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        redis = _FakeRedis()
        token = mint_external_token(SECRET, "fm-worker-0")
        meta = {"hardware_json": _hardware_json("GPU-aaa")}

        before, _ = _build_servicer(redis=redis)
        await _register(before, token, meta=meta)
        after, _ = _build_servicer(redis=redis)
        await _register(after, token, meta=meta)

        info = after._worker_manager.get_worker_info("fm-worker-0")
        assert info is not None and info.hardware is not None


class TestRegisterWorkerRecordsVerifiedAlias:
    """The recorded alias is the name the supervisor can verify, never the
    self-reported one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reported", [None, "fm-worker-0"])
    async def test_external_alias_is_token_name(
        self, monkeypatch: pytest.MonkeyPatch, reported: str | None
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        worker_id = await _register(servicer, token, alias=reported)

        assert redis.hashes[worker_key(worker_id)]["alias"] == "fm-worker-0"

    @pytest.mark.asyncio
    async def test_mismatched_alias_is_overridden_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)
        servicer, redis = _build_servicer()
        token = mint_external_token(SECRET, "fm-worker-0")

        with caplog.at_level(logging.WARNING):
            worker_id = await _register(servicer, token, alias="someone-else")

        assert redis.hashes[worker_key(worker_id)]["alias"] == "fm-worker-0"
        assert "'someone-else'" in caplog.text

    @pytest.mark.asyncio
    async def test_managed_alias_is_supervisor_name(self) -> None:
        servicer, redis = _build_servicer()
        token = servicer._registry.new_token()
        worker = ExternalWorkerFactory(system_principal=None).create_worker(  # type: ignore[arg-type]
            token, ExternalWorkerConfig(), alias="flowmesh_server_worker_cpu_0"
        )
        servicer._registry.add(worker)

        worker_id = await _register(servicer, token, alias="3f9a1c0e7b2d4a55")

        assert (
            redis.hashes[worker_key(worker_id)]["alias"]
            == "flowmesh_server_worker_cpu_0"
        )


def test_worker_environment_carries_name_as_alias() -> None:
    principal = PrincipalContext(
        principal_id="system",
        org_id="org",
        external_id="system",
        principal_type="user",
        scopes=[],
    )
    worker = ExternalWorkerFactory(system_principal=principal).create_worker(
        WorkerRegistry().new_token(),
        ExternalWorkerConfig(worker_alias="requested"),
        alias="resolved-name",
    )
    assert worker._base_environment()["WORKER_ALIAS"] == "resolved-name"


def _host_pool(n: int) -> ResourceManager:
    """A host with GPUs 0..n-1 whose UUIDs are `GPU-<index>`."""
    rm = object.__new__(ResourceManager)
    rm._env = MachineEnv(
        cpu_count=16,
        gpu_families={i: GpuArch.HOPPER for i in range(n)},
        available_gpus=set(range(n)),
        gpu_uuids={f"GPU-{i}": i for i in range(n)},
    )
    return rm


async def _push_events(
    servicer: SupervisorServicer, token: str, *payloads: dict[str, Any]
) -> None:
    async def messages() -> AsyncIterator[supervisor_pb2.EventMessage]:
        for payload in payloads:
            message = supervisor_pb2.EventMessage()
            message.payload.update(payload)
            yield message

    await servicer.PushEvents(messages(), cast(Any, _FakeContext(token)))


class TestExternalGpuHolds:
    """An external worker on the supervisor's host holds its GPUs out of the
    pool until it unregisters or is destroyed -- never because it crashed."""

    TOKEN = mint_external_token(SECRET, "fm-worker-0")

    @pytest.fixture(autouse=True)
    def _secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", SECRET)

    def _servicer(
        self, rm: ResourceManager | None, calls: list[int] | None = None
    ) -> SupervisorServicer:
        callback = None if calls is None else (lambda: calls.append(1))
        servicer, _ = _build_servicer(
            resource_manager=rm, capacity_change_callback=callback
        )
        return servicer

    async def _register(self, servicer: SupervisorServicer, *uuids: str) -> str:
        return await _register(
            servicer, self.TOKEN, meta={"hardware_json": _hardware_json(*uuids)}
        )

    def _held(self, servicer: SupervisorServicer) -> list[int]:
        worker = servicer._registry.try_get(cast(Any, self.TOKEN))
        assert isinstance(worker, ExternalWorkerAdapter)
        return worker.held_gpus

    @pytest.mark.asyncio
    async def test_registration_holds_the_reported_host_gpus(self) -> None:
        rm, calls = _host_pool(4), list[int]()
        servicer = self._servicer(rm, calls)

        await self._register(servicer, "GPU-1", "GPU-2")

        assert rm._env.available_gpus == {0, 3}
        assert self._held(servicer) == [1, 2]
        assert len(calls) == 2  # admission, then the claim
        info = servicer._worker_manager.get_worker_info("fm-worker-0")
        assert info is not None and info.held_gpus == [1, 2]

    @pytest.mark.asyncio
    async def test_node_worker_listing_carries_held_gpus(self) -> None:
        """GET_WORKERS dumps each WorkerInfo; the root re-validates it as
        NodeWorkerInfo, which must keep the field."""
        servicer = self._servicer(_host_pool(2))
        await self._register(servicer, "GPU-1")

        [info] = servicer._worker_manager.list_workers()
        node_info = NodeWorkerInfo.model_validate(
            info.model_dump() | {"node_id": "nde-1", "status": "IDLE"}
        )

        assert node_info.held_gpus == [1]

    @pytest.mark.asyncio
    async def test_reregistering_the_same_gpus_changes_nothing(self) -> None:
        rm, calls = _host_pool(4), list[int]()
        servicer = self._servicer(rm, calls)
        await self._register(servicer, "GPU-1")
        before = len(calls)

        await self._register(servicer, "GPU-1")

        assert rm._env.available_gpus == {0, 2, 3}
        assert len(calls) == before

    @pytest.mark.asyncio
    async def test_reregistering_other_gpus_moves_the_hold(self) -> None:
        rm = _host_pool(4)
        servicer = self._servicer(rm)
        await self._register(servicer, "GPU-1")

        await self._register(servicer, "GPU-3")

        assert rm._env.available_gpus == {0, 1, 2}
        assert self._held(servicer) == [3]

    @pytest.mark.asyncio
    async def test_gpus_of_another_host_are_not_held(self) -> None:
        """A remote worker's GPUs have UUIDs this host's pool never lists."""
        rm, calls = _host_pool(2), list[int]()
        servicer = self._servicer(rm, calls)

        await self._register(servicer, "GPU-elsewhere")

        assert rm.available_gpu_count() == 2
        assert len(calls) == 1  # admission only
        info = servicer._worker_manager.get_worker_info("fm-worker-0")
        assert info is not None and info.held_gpus == []

    @pytest.mark.asyncio
    async def test_worker_sent_unregister_releases(self) -> None:
        rm, calls = _host_pool(2), list[int]()
        servicer = self._servicer(rm, calls)
        worker_id = await self._register(servicer, "GPU-0")
        before = len(calls)

        await _push_events(
            servicer,
            self.TOKEN,
            {"type": "REGISTER", "worker_id": worker_id},
            {"type": "UNREGISTER", "worker_id": worker_id},
        )

        assert rm.available_gpu_count() == 2
        assert len(calls) == before + 1

    @pytest.mark.asyncio
    async def test_stream_ending_without_unregister_keeps_the_hold(self) -> None:
        """A crash closes the stream; the supervisor fabricates an UNREGISTER
        for the server, but the cards stay held for the restarted worker."""
        rm = _host_pool(2)
        servicer = self._servicer(rm)
        worker_id = await self._register(servicer, "GPU-0")

        await _push_events(
            servicer, self.TOKEN, {"type": "REGISTER", "worker_id": worker_id}
        )

        relayed = cast(_FakeRelayService, servicer._relay_service).events
        assert relayed[-1]["type"] == "UNREGISTER"
        assert rm._env.available_gpus == {1}
        assert self._held(servicer) == [0]

    @pytest.mark.asyncio
    async def test_unregister_of_a_superseded_registration_keeps_the_hold(
        self,
    ) -> None:
        rm = _host_pool(2)
        servicer = self._servicer(rm)
        await self._register(servicer, "GPU-0")

        await _push_events(
            servicer, self.TOKEN, {"type": "UNREGISTER", "worker_id": "wkr-stale"}
        )

        assert rm._env.available_gpus == {1}

    @pytest.mark.asyncio
    async def test_unregister_on_an_old_stream_releases_the_current_adapter(
        self,
    ) -> None:
        """After a destroy and re-admission, the worker's graceful UNREGISTER
        can arrive on a stream opened for the destroyed adapter. It carries the
        current id, so the current adapter's holds are released."""
        rm = _host_pool(2)
        servicer = self._servicer(rm)
        await self._register(servicer, "GPU-0")
        opened, send = asyncio.Event(), asyncio.Event()
        payload: dict[str, Any] = {}

        async def messages() -> AsyncIterator[supervisor_pb2.EventMessage]:
            opened.set()
            await send.wait()
            message = supervisor_pb2.EventMessage()
            message.payload.update(payload)
            yield message

        # The stream opens -- and resolves its adapter -- before the destroy.
        stream = asyncio.create_task(
            servicer.PushEvents(messages(), cast(Any, _FakeContext(self.TOKEN)))
        )
        await asyncio.wait_for(opened.wait(), timeout=5)
        await servicer._worker_manager.destroy_worker("fm-worker-0")
        worker_id = await self._register(servicer, "GPU-1")
        payload.update(type="UNREGISTER", worker_id=worker_id)
        assert rm._env.available_gpus == {0}

        send.set()
        await asyncio.wait_for(stream, timeout=5)

        assert servicer._registry.get_worker_id(cast(Any, self.TOKEN)) == worker_id
        assert rm._env.available_gpus == {0, 1}

    @pytest.mark.asyncio
    async def test_destroy_releases_and_is_idempotent(self) -> None:
        rm = _host_pool(2)
        servicer = self._servicer(rm)
        await self._register(servicer, "GPU-0")
        worker = servicer._registry.try_get(cast(Any, self.TOKEN))
        assert worker is not None

        await servicer._worker_manager.destroy_worker("fm-worker-0")
        factory = servicer._worker_manager._providers["external"].factory
        factory.destroy_worker(worker)

        assert rm.available_gpu_count() == 2

    @pytest.mark.asyncio
    async def test_supervisor_shutdown_releases(self) -> None:
        rm = _host_pool(2)
        servicer = self._servicer(rm)
        await self._register(servicer, "GPU-0", "GPU-1")

        await servicer._worker_manager.stop()

        assert rm.available_gpu_count() == 2

    @pytest.mark.asyncio
    async def test_a_card_another_worker_holds_is_shared_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rm = _host_pool(2)
        rm.reserve_gpus(devices=[0])
        servicer = self._servicer(rm)

        with caplog.at_level(logging.WARNING, logger="supervisor"):
            await self._register(servicer, "GPU-0", "GPU-1")

        assert self._held(servicer) == [0, 1]
        assert "already holds" in caplog.text
        # The other holder's release leaves the card held by this worker.
        rm.deallocate_gpus([0])
        assert rm.available_gpu_count() == 0

    @pytest.mark.asyncio
    async def test_dockerless_host_admits_without_holding(self) -> None:
        servicer = self._servicer(None)

        worker_id = await self._register(servicer, "GPU-0")

        assert worker_id
        assert self._held(servicer) == []
