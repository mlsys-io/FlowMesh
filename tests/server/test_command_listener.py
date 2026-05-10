"""Regression tests for CommandListener handler error paths.

Focus: malformed or missing payloads must return CommandResponse.error and
must never raise out of the handler (which would kill the listener thread).
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from server.supervisor.services.command_listener import CommandListener
from shared.schemas.command import (
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
        info.name = "w-test"
        info.model_dump = MagicMock(return_value={"name": "w-test"})
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
        assert resp.data["name"] == "w-test"
        init_config = self.cl._wm.create_worker.call_args[0][0]
        assert init_config.provider == "docker"
        assert init_config.worker_config["worker_alias"] == "my-worker"

    def test_invalid_payload_returns_error(self) -> None:
        resp = self._handle(None)
        assert not resp.success


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
        info.name = "w-cpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0"})

        assert resp.success
        call_args = self.cl._wm.create_worker.call_args[0][0]
        assert "cuda_devices" not in call_args.worker_config

    # --- GPU workers (handler forwards gpu_count; factory reserves) ---

    def test_two_gpu_worker_sets_type_and_gpu_count(self) -> None:
        info = MagicMock()
        info.name = "w-gpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "2", "worker_alias": "my-worker"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["worker_type"] == "gpu"
        assert cfg["gpu_count"] == 2
        assert "cuda_devices" not in cfg

    def test_four_gpu_worker(self) -> None:
        info = MagicMock()
        info.name = "w-gpu"
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
        info.name = "w-gpu"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "2", "cuda_devices": [2, 3]})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["cuda_devices"] == [2, 3]
        assert "gpu_count" not in cfg

    # --- Worker alias ---

    def test_alias_auto_generated_with_worker_prefix(self) -> None:
        info = MagicMock()
        info.name = "w-test"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        alias = cfg.get("worker_alias", "")
        assert alias.startswith("worker_cpu_"), f"unexpected alias: {alias!r}"

    def test_explicit_alias_preserved(self) -> None:
        info = MagicMock()
        info.name = "w-test"
        self.cl._wm.create_worker = AsyncMock(return_value=info)  # type: ignore[method-assign]

        resp = self._handle({"gpu_count": "0", "worker_alias": "my-alias"})

        assert resp.success
        cfg = self.cl._wm.create_worker.call_args[0][0].worker_config
        assert cfg["worker_alias"] == "my-alias"

    def test_alias_unique_across_calls(self) -> None:
        info = MagicMock()
        info.name = "w-test"
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
        assert "worker_name" in (resp.message or "").lower()

    def test_missing_worker_name_returns_error(self) -> None:
        resp = self._handle({})
        assert not resp.success

    def test_empty_worker_name_returns_error(self) -> None:
        resp = self._handle({"worker_name": ""})
        assert not resp.success

    def test_valid_worker_name_calls_destroy(self) -> None:
        self.cl._wm.destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        resp = self._handle({"worker_name": "worker-abc123"})

        assert resp.success
        self.cl._wm.destroy_worker.assert_called_once_with("worker-abc123")


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
                _cmd(CommandType.START_WORKER, {"worker_name": f"w-{i}"})
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
                cl._dispatch(_cmd(CommandType.STOP_WORKER, {"worker_name": "w-1"}))
            )
            await asyncio.sleep(0.05)
            destroy_task = asyncio.create_task(
                cl._dispatch(_cmd(CommandType.DESTROY_WORKER, {"worker_name": "w-1"}))
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
                _cmd(CommandType.DESTROY_WORKER, {"worker_name": "w-gone"})
            )

        asyncio.run(go())
        assert "w-gone" not in cl._worker_locks

    def test_destroy_workers_duplicate_names_does_not_deadlock(self) -> None:
        cl = self._setup()
        cl._wm.destroy_workers = AsyncMock(return_value=None)  # type: ignore[method-assign]

        async def go() -> CommandResponse:
            return await asyncio.wait_for(
                cl._dispatch(
                    _cmd(
                        CommandType.DESTROY_WORKERS,
                        {"worker_names": ["w-1", "w-1", "w-2", "w-2"]},
                    )
                ),
                timeout=2.0,
            )

        resp = asyncio.run(go())
        assert resp.success


# ------------------------------------------------------------------ #
# Target worker name resolution
# ------------------------------------------------------------------ #


class TestTargetWorkerNames:
    def test_single_worker_commands(self) -> None:
        for cmd_type in (
            CommandType.START_WORKER,
            CommandType.STOP_WORKER,
            CommandType.DESTROY_WORKER,
        ):
            assert CommandListener._target_worker_names(
                _cmd(cmd_type, {"worker_name": "w-1"})
            ) == ["w-1"]

    def test_destroy_workers_sorts_names(self) -> None:
        names = CommandListener._target_worker_names(
            _cmd(CommandType.DESTROY_WORKERS, {"worker_names": ["w-3", "w-1", "w-2"]})
        )
        assert names == ["w-1", "w-2", "w-3"]

    def test_destroy_workers_dedupes_names(self) -> None:
        names = CommandListener._target_worker_names(
            _cmd(
                CommandType.DESTROY_WORKERS,
                {"worker_names": ["w-2", "w-1", "w-2", "w-1", "w-3"]},
            )
        )
        assert names == ["w-1", "w-2", "w-3"]

    def test_destroy_workers_no_names(self) -> None:
        assert (
            CommandListener._target_worker_names(_cmd(CommandType.DESTROY_WORKERS, {}))
            == []
        )

    def test_create_and_get_skip_locks(self) -> None:
        for cmd_type in (
            CommandType.CREATE_WORKER,
            CommandType.CREATE_WORKER_ON_NODE,
            CommandType.GET_WORKERS,
            CommandType.START_SSH_RELAY,
        ):
            payload = {"worker_name": "w-1"}
            assert CommandListener._target_worker_names(_cmd(cmd_type, payload)) == []
