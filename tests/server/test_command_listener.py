"""Regression tests for CommandListener handler error paths.

Focus: malformed or missing payloads must return CommandResponse.error and
must never raise out of the handler (which would kill the listener thread).
"""

import asyncio
import logging
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from server.supervisor.adapters.docker import DockerWorkerConfig
from server.supervisor.manager import ManagerNotStartedError, ProviderUnavailableError
from server.supervisor.services.command_listener import CommandListener
from shared.schemas.command import (
    CommandErrorCode,
    CommandMessage,
    CommandResponse,
    CommandType,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _listener() -> CommandListener:
    """Build a CommandListener with stub dependencies."""
    return CommandListener(
        redis=MagicMock(),
        node_id="test-server",
        worker_manager=MagicMock(),
        logger=logging.getLogger("test-cl"),
    )


def _cmd(command: CommandType, payload: dict | None = None) -> CommandMessage:
    return CommandMessage(command=command, payload=payload)


def _run(coro: object) -> CommandResponse:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# CREATE_WORKER — pass-through to WorkerManager
# ------------------------------------------------------------------ #


class TestHandleCreateWorkerCmd:
    def setup_method(self) -> None:
        self.cl = _listener()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.CREATE_WORKER, payload)
        return _run(self.cl._handle_create_worker_cmd(cmd))

    def test_valid_init_config(self) -> None:
        info = MagicMock()
        info.alias = "w-test"
        info.model_dump = MagicMock(return_value={"alias": "w-test"})
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle(
            {
                "provider": "docker",
                "init_on_start": True,
                "worker_config": {"worker_alias": "my-worker", "worker_type": "cpu"},
            }
        )

        assert resp.success
        assert resp.data is not None
        assert resp.data["alias"] == "w-test"
        init_config = self.cl._wm.create_worker.call_args[0][0]
        assert init_config.provider == "docker"
        assert init_config.worker_config["worker_alias"] == "my-worker"

    def test_invalid_payload_returns_error(self) -> None:
        resp = self._handle(None)
        assert not resp.success
        assert resp.error_code == CommandErrorCode.INVALID_PAYLOAD

    def test_invalid_worker_config_sets_invalid_payload(self) -> None:
        """CREATE_WORKER is the one command whose payload the API caller
        supplies, so a config that fails validation is the caller's error."""
        with pytest.raises(ValidationError) as excinfo:
            DockerWorkerConfig.model_validate({"worker_type": "banana"})
        self.cl._wm.create_worker = AsyncMock(  # type: ignore[method-assign]
            side_effect=excinfo.value
        )
        resp = self._handle(
            {"provider": "docker", "worker_config": {"worker_type": "banana"}}
        )
        assert not resp.success
        assert resp.error_code == CommandErrorCode.INVALID_PAYLOAD

    def test_manager_not_started_sets_not_ready(self) -> None:
        self.cl._wm.create_worker = AsyncMock(  # type: ignore[method-assign]
            side_effect=ManagerNotStartedError("WorkerManager not started")
        )
        resp = self._handle({"provider": "docker"})
        assert not resp.success
        assert resp.error_code == CommandErrorCode.NOT_READY

    def test_provider_unavailable_sets_error_code(self) -> None:
        self.cl._wm.create_worker = AsyncMock(  # type: ignore[method-assign]
            side_effect=ProviderUnavailableError(
                "Worker provider 'docker' is not available on this node; "
                "available providers: external"
            )
        )
        resp = self._handle({"provider": "docker"})
        assert not resp.success
        assert resp.error_code == CommandErrorCode.PROVIDER_UNAVAILABLE
        assert "docker" in (resp.message or "")
        assert "external" in (resp.message or "")


# ------------------------------------------------------------------ #
# GET_PROVIDERS
# ------------------------------------------------------------------ #


class TestHandleGetProvidersCmd:
    def setup_method(self) -> None:
        self.cl = _listener()

    def test_returns_providers(self) -> None:
        self.cl._wm.available_providers = MagicMock(  # type: ignore[method-assign]
            return_value=["docker", "external"]
        )
        cmd = _cmd(CommandType.GET_PROVIDERS)
        resp = self.cl._handle_get_providers_cmd(cmd)
        assert resp.success
        assert resp.data == {"providers": ["docker", "external"]}


# ------------------------------------------------------------------ #
# CREATE_WORKER_ON_NODE — flat Docker payload with GPU allocation
# ------------------------------------------------------------------ #


class TestHandleCreateWorkerOnNodeCmd:
    def setup_method(self) -> None:
        self.cl = _listener()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.CREATE_WORKER_ON_NODE, payload)
        return _run(self.cl._handle_create_worker_on_node_cmd(cmd))

    # --- Malformed payloads ---

    def test_none_payload_returns_error_not_raises(self) -> None:
        resp = self._handle(None)
        assert not resp.success

    def test_non_integer_gpu_count_returns_error(self) -> None:
        resp = self._handle({"gpu_count": "not-a-number"})
        assert not resp.success

    def test_empty_string_gpu_count_returns_error(self) -> None:
        resp = self._handle({"gpu_count": ""})
        assert not resp.success

    def test_none_gpu_count_returns_error(self) -> None:
        resp = self._handle({"gpu_count": None})
        assert not resp.success

    # --- CPU worker ---

    def test_zero_gpu_count_creates_cpu_worker(self) -> None:
        info = MagicMock()
        info.alias = "w-cpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0"})

        assert resp.success
        call_args = self.cl._wm.create_worker.call_args[0][0]
        assert "cuda_devices" not in call_args.worker_config

    # --- GPU workers (handler forwards gpu_count; factory reserves) ---

    def test_two_gpu_worker_sets_type_and_gpu_count(self) -> None:
        info = MagicMock()
        info.alias = "w-gpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "2", "worker_alias": "my-worker"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["worker_type"] == "gpu"
        assert cfg["gpu_count"] == 2
        assert "cuda_devices" not in cfg

    def test_four_gpu_worker(self) -> None:
        info = MagicMock()
        info.alias = "w-gpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "4"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["worker_type"] == "gpu"
        assert cfg["gpu_count"] == 4
        assert "cuda_devices" not in cfg

    # --- Reservation failures surface via WorkerManager ---

    def test_reserve_failure_surfaces_as_error(self) -> None:
        self.cl._wm.create_worker = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError("Not enough available GPUs")
        )

        resp = self._handle({"gpu_count": "1"})

        assert not resp.success
        assert "Not enough available GPUs" in (resp.message or "")

    # --- Validation: worker_type vs gpu_count ---

    def test_invalid_worker_type_for_gpu_count_returns_error(self) -> None:
        resp = self._handle({"gpu_count": "2", "worker_type": "cpu"})

        assert not resp.success
        assert "Invalid worker_type" in (resp.message or "")

    # --- Validation: cuda_devices vs gpu_count ---

    def test_cuda_devices_length_mismatch_returns_error(self) -> None:
        resp = self._handle({"gpu_count": "2", "cuda_devices": [0]})

        assert not resp.success
        assert "must match gpu_count" in (resp.message or "")

    def test_explicit_cuda_devices_passed_through(self) -> None:
        info = MagicMock()
        info.alias = "w-gpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "2", "cuda_devices": [2, 3]})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["cuda_devices"] == [2, 3]
        assert "gpu_count" not in cfg

    # --- Worker alias ---

    def test_alias_auto_generated_with_worker_prefix(self) -> None:
        info = MagicMock()
        info.alias = "w-test"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        alias = cfg.get("worker_alias", "")
        assert alias.startswith("worker_cpu_"), f"unexpected alias: {alias!r}"

    def test_explicit_alias_preserved(self) -> None:
        info = MagicMock()
        info.alias = "w-test"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0", "worker_alias": "my-alias"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["worker_alias"] == "my-alias"

    def test_alias_unique_across_calls(self) -> None:
        info = MagicMock()
        info.alias = "w-test"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        self._handle({"gpu_count": "0"})
        alias1 = self.cl._wm.create_worker.call_args[0][0].worker_config["worker_alias"]

        self._handle({"gpu_count": "0"})
        alias2 = self.cl._wm.create_worker.call_args[0][0].worker_config["worker_alias"]

        assert alias1 != alias2


# ------------------------------------------------------------------ #
# DESTROY_WORKER — malformed payload
# ------------------------------------------------------------------ #


class TestHandleDestroyWorkerCmd:
    def setup_method(self) -> None:
        self.cl = _listener()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.DESTROY_WORKER, payload)
        return _run(self.cl._handle_destroy_worker_cmd(cmd))

    def test_none_payload_returns_error(self) -> None:
        resp = self._handle(None)
        assert not resp.success
        assert "worker_alias" in (resp.message or "").lower()

    def test_missing_worker_alias_returns_error(self) -> None:
        resp = self._handle({})
        assert not resp.success

    def test_empty_worker_alias_returns_error(self) -> None:
        resp = self._handle({"worker_alias": ""})
        assert not resp.success

    def test_valid_worker_alias_calls_destroy(self) -> None:
        self.cl._wm.destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        resp = self._handle({"worker_alias": "worker-abc123"})

        assert resp.success
        self.cl._wm.destroy_worker.assert_called_once_with("worker-abc123")


class TestLegacyRootPayloads:
    """A root one release behind sends `worker_name(s)` and reads `name`."""

    def setup_method(self) -> None:
        self.cl = _listener()

    def test_start_and_stop_accept_worker_name(self) -> None:
        self.cl._wm.start_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        self.cl._wm.stop_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        start = _cmd(CommandType.START_WORKER, {"worker_name": "w-1"})
        stop = _cmd(CommandType.STOP_WORKER, {"worker_name": "w-1"})

        assert _run(self.cl._handle_start_worker_cmd(start)).success
        assert _run(self.cl._handle_stop_worker_cmd(stop)).success
        self.cl._wm.start_worker.assert_called_once_with("w-1")
        self.cl._wm.stop_worker.assert_called_once_with("w-1")

    def test_destroy_workers_accepts_worker_names(self) -> None:
        self.cl._wm.destroy_workers = AsyncMock()  # type: ignore[method-assign]

        cmd = _cmd(CommandType.DESTROY_WORKERS, {"worker_names": ["w-1", "w-2"]})

        assert _run(self.cl._handle_destroy_workers_cmd(cmd)).success
        self.cl._wm.destroy_workers.assert_called_once_with({"w-1", "w-2"})

    def test_destroy_worker_accepts_worker_name(self) -> None:
        self.cl._wm.destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        cmd = _cmd(CommandType.DESTROY_WORKER, {"worker_name": "w-1"})

        assert _run(self.cl._handle_destroy_worker_cmd(cmd)).success
        self.cl._wm.destroy_worker.assert_called_once_with("w-1")

    def test_get_single_worker_accepts_worker_name(self) -> None:
        info = MagicMock()
        info.alias = "w-1"
        info.model_dump = MagicMock(return_value={"alias": "w-1"})
        self.cl._wm.get_worker_info = MagicMock(return_value=info)  # type: ignore[method-assign]

        cmd = _cmd(CommandType.GET_WORKERS, {"worker_name": "w-1"})
        resp = self.cl._handle_get_workers_cmd(cmd)

        assert resp.data == {"workers": [{"alias": "w-1", "name": "w-1"}]}
        self.cl._wm.get_worker_info.assert_called_once_with("w-1")

    def test_get_workers_reports_alias_as_name(self) -> None:
        info = MagicMock()
        info.alias = "w-1"
        info.model_dump = MagicMock(return_value={"alias": "w-1"})
        self.cl._wm.list_workers = MagicMock(return_value=[info])  # type: ignore[method-assign]

        resp = self.cl._handle_get_workers_cmd(_cmd(CommandType.GET_WORKERS))

        assert resp.data == {"workers": [{"alias": "w-1", "name": "w-1"}]}


# ------------------------------------------------------------------ #
# Parallel dispatch — different workers run concurrently; same-worker
# commands serialize via per-worker locks.
# ------------------------------------------------------------------ #


class TestParallelDispatch:
    def _setup(self) -> CommandListener:
        cl = _listener()
        cl._sem = asyncio.Semaphore(32)
        return cl

    def test_distinct_workers_run_concurrently(self) -> None:
        cl = self._setup()

        async def slow_start(name: str) -> bool:
            await asyncio.sleep(0.2)
            return True

        cl._wm.start_worker = AsyncMock(side_effect=slow_start)  # type: ignore[method-assign]

        async def go() -> tuple[float, list[CommandResponse]]:
            cmds = [
                _cmd(CommandType.START_WORKER, {"worker_alias": f"w-{i}"})
                for i in range(4)
            ]
            t0 = asyncio.get_event_loop().time()
            results = await asyncio.gather(*(cl._dispatch(c) for c in cmds))
            return asyncio.get_event_loop().time() - t0, results

        elapsed, results = asyncio.run(go())
        assert all(r.success for r in results)
        # Sequential would be ~0.8s; parallel should be ~0.2s. Pad for CI.
        assert elapsed < 0.6, f"dispatch did not parallelize (elapsed={elapsed:.2f}s)"

    def test_same_worker_serializes(self) -> None:
        cl = self._setup()

        order: list[str] = []
        gate = asyncio.Event()

        async def slow_stop(name: str) -> bool:
            order.append(f"stop-start-{name}")
            await gate.wait()
            order.append(f"stop-end-{name}")
            return True

        async def fast_destroy(name: str) -> bool:
            order.append(f"destroy-start-{name}")
            order.append(f"destroy-end-{name}")
            return True

        cl._wm.stop_worker = AsyncMock(side_effect=slow_stop)  # type: ignore[method-assign]
        cl._wm.destroy_worker = AsyncMock(side_effect=fast_destroy)  # type: ignore[method-assign]

        async def go() -> tuple[CommandResponse, CommandResponse]:
            stop_task = asyncio.create_task(
                cl._dispatch(_cmd(CommandType.STOP_WORKER, {"worker_alias": "w-1"}))
            )
            await asyncio.sleep(0.05)
            destroy_task = asyncio.create_task(
                cl._dispatch(_cmd(CommandType.DESTROY_WORKER, {"worker_alias": "w-1"}))
            )
            # Destroy must NOT have started while stop is blocked.
            await asyncio.sleep(0.05)
            assert "destroy-start-w-1" not in order
            gate.set()
            return await asyncio.gather(stop_task, destroy_task)

        results = asyncio.run(go())
        assert all(r.success for r in results)
        assert order == [
            "stop-start-w-1",
            "stop-end-w-1",
            "destroy-start-w-1",
            "destroy-end-w-1",
        ]

    def test_destroy_clears_worker_lock(self) -> None:
        cl = self._setup()
        cl._wm.destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        async def go() -> None:
            await cl._dispatch(
                _cmd(CommandType.DESTROY_WORKER, {"worker_alias": "w-gone"})
            )

        asyncio.run(go())
        assert "w-gone" not in cl._worker_locks

    def test_drain_does_not_block_event_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: stop() runs on the supervisor loop. _drain_inflight()
        must await cooperatively so the inflight tasks (also on this loop)
        keep being scheduled. A blocking concurrent.futures.wait() here would
        starve them and force a cancellation at the drain timeout."""
        # Short drain timeout so a regression fails fast instead of hanging
        # for the full 10s default.
        monkeypatch.setattr(
            "server.supervisor.services.command_listener._STOP_DRAIN_TIMEOUT", 0.5
        )

        cl = self._setup()

        async def slow_start(name: str) -> bool:
            await asyncio.sleep(0.05)
            return True

        cl._wm.start_worker = AsyncMock(side_effect=slow_start)  # type: ignore[method-assign]

        async def go() -> bool:
            loop = asyncio.get_running_loop()
            cl._loop = loop
            cmd_obj = _cmd(CommandType.START_WORKER, {"worker_alias": "w-1"})
            submitted = threading.Event()
            captured: list = []

            def submit_from_producer() -> None:
                fut = asyncio.run_coroutine_threadsafe(cl._dispatch(cmd_obj), loop)
                cl._inflight.add(fut)
                fut.add_done_callback(cl._make_done_callback(cmd_obj, lambda _r: None))
                captured.append(fut)
                submitted.set()

            threading.Thread(target=submit_from_producer, daemon=True).start()
            await asyncio.to_thread(submitted.wait)

            await cl._drain_inflight()
            fut = captured[0]
            return fut.done() and not fut.cancelled()

        completed = asyncio.run(go())
        assert completed, "inflight task was cancelled — drain blocked the loop"

    def test_destroy_workers_duplicate_names_does_not_deadlock(self) -> None:
        cl = self._setup()
        cl._wm.destroy_workers = AsyncMock(return_value=None)  # type: ignore[method-assign]

        async def go() -> CommandResponse:
            return await asyncio.wait_for(
                cl._dispatch(
                    _cmd(
                        CommandType.DESTROY_WORKERS,
                        {"worker_aliases": ["w-1", "w-1", "w-2", "w-2"]},
                    )
                ),
                timeout=2.0,
            )

        resp = asyncio.run(go())
        assert resp.success


# ------------------------------------------------------------------ #
# Target worker name resolution
# ------------------------------------------------------------------ #


class TestTargetWorkerAliases:
    def test_single_worker_commands(self) -> None:
        for cmd_type in (
            CommandType.START_WORKER,
            CommandType.STOP_WORKER,
            CommandType.DESTROY_WORKER,
        ):
            assert CommandListener._target_worker_aliases(
                _cmd(cmd_type, {"worker_alias": "w-1"})
            ) == ["w-1"]

    def test_destroy_workers_sorts_aliases(self) -> None:
        aliases = CommandListener._target_worker_aliases(
            _cmd(CommandType.DESTROY_WORKERS, {"worker_aliases": ["w-3", "w-1", "w-2"]})
        )
        assert aliases == ["w-1", "w-2", "w-3"]

    def test_destroy_workers_dedupes_aliases(self) -> None:
        aliases = CommandListener._target_worker_aliases(
            _cmd(
                CommandType.DESTROY_WORKERS,
                {"worker_aliases": ["w-2", "w-1", "w-2", "w-1", "w-3"]},
            )
        )
        assert aliases == ["w-1", "w-2", "w-3"]

    def test_destroy_workers_no_aliases(self) -> None:
        assert (
            CommandListener._target_worker_aliases(
                _cmd(CommandType.DESTROY_WORKERS, {})
            )
            == []
        )

    def test_create_and_get_skip_locks(self) -> None:
        for cmd_type in (
            CommandType.CREATE_WORKER,
            CommandType.CREATE_WORKER_ON_NODE,
            CommandType.GET_WORKERS,
            CommandType.START_RELAY,
        ):
            payload = {"worker_alias": "w-1"}
            assert CommandListener._target_worker_aliases(_cmd(cmd_type, payload)) == []

    def test_legacy_payload_keys(self) -> None:
        assert CommandListener._target_worker_aliases(
            _cmd(CommandType.STOP_WORKER, {"worker_name": "w-1"})
        ) == ["w-1"]
        assert CommandListener._target_worker_aliases(
            _cmd(CommandType.DESTROY_WORKERS, {"worker_names": ["w-2", "w-1"]})
        ) == ["w-1", "w-2"]
