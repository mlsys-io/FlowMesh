"""Regression tests for CommandListener handler error paths.

Focus: malformed or missing payloads must return CommandResponse.error and
must never raise out of the handler (which would kill the listener thread).
"""

import asyncio
import logging
import threading
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


class _RunningLoop:
    """Event loop running in a background thread — required because handlers
    use asyncio.run_coroutine_threadsafe(...).result() which needs a live loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


# ------------------------------------------------------------------ #
# CREATE_WORKER — pass-through to WorkerManager
# ------------------------------------------------------------------ #


class TestHandleCreateWorkerCmd:
    def setup_method(self) -> None:
        self.cl = _listener()
        self._rl = _RunningLoop()

    def teardown_method(self) -> None:
        self._rl.close()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.CREATE_WORKER, payload)
        return self.cl._handle_create_worker_cmd(cmd, self._rl.loop)

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
        self._rl = _RunningLoop()

    def teardown_method(self) -> None:
        self._rl.close()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.CREATE_WORKER_ON_NODE, payload)
        return self.cl._handle_create_worker_on_node_cmd(cmd, self._rl.loop)

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
        self._rl = _RunningLoop()

    def teardown_method(self) -> None:
        self._rl.close()

    def _handle(self, payload: dict | None) -> CommandResponse:
        cmd = _cmd(CommandType.DESTROY_WORKER, payload)
        return self.cl._handle_destroy_worker_cmd(cmd, self._rl.loop)

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
