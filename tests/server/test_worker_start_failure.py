"""Tests for how the worker manager reports a worker that fails to start or is
torn down."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.supervisor.adapters.base import WorkerAdapter
from server.supervisor.manager import WorkerInitConfig
from server.supervisor.schemas import WorkerStatus
from tests.server.supervisor_helpers import StubWorkerManager


def _worker(*, started: bool) -> MagicMock:
    worker = MagicMock(spec=WorkerAdapter)
    worker.alias = "gpu_0"
    worker.token = "gpu_0.token"
    worker.status = WorkerStatus.STOPPED
    worker.start = AsyncMock(return_value=started)
    return worker


def _info_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


class TestStartWorkerFailure:
    @pytest.mark.asyncio
    async def test_failed_start_is_logged_as_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=False)

        with caplog.at_level(logging.ERROR, logger="test.supervisor"):
            result = await wm._start_worker(worker)

        assert result is False
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == ["Worker gpu_0 failed to start"]

    @pytest.mark.asyncio
    async def test_failed_start_keeps_the_worker_registered(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=False)

        assert await wm._start_worker(worker) is False
        wm._stop_and_destroy_worker.assert_not_awaited()
        registry.try_pop.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_start_logs_no_error_and_keeps_the_worker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=True)

        with caplog.at_level(logging.ERROR, logger="test.supervisor"):
            result = await wm._start_worker(worker)

        assert result is True
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        wm._stop_and_destroy_worker.assert_not_awaited()
        registry.try_pop.assert_not_called()


class TestCreateWorkerFailure:
    @pytest.mark.asyncio
    async def test_failed_create_unwinds_the_worker_it_made(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        worker = _worker(started=False)
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Failed to start worker 'gpu_0'"):
            await wm.create_worker(WorkerInitConfig(init_on_start=True))

        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)

    @pytest.mark.asyncio
    async def test_create_whose_start_raises_unwinds_the_worker_it_made(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        worker = _worker(started=False)
        worker.start = AsyncMock(side_effect=OSError("docker unavailable"))
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with pytest.raises(OSError, match="docker unavailable"):
            await wm.create_worker(WorkerInitConfig(init_on_start=True))

        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)


class TestStopAndDestroyWorkerLog:
    @pytest.mark.asyncio
    async def test_a_worker_that_failed_to_start_is_logged_as_destroyed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        wm._destroy_worker = MagicMock()  # type: ignore[method-assign]
        worker = _worker(started=False)
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO, logger="test.supervisor"):
            with pytest.raises(RuntimeError):
                await wm.create_worker(WorkerInitConfig(init_on_start=True))

        assert _info_messages(caplog) == [
            "Destroying worker gpu_0 that is not running.",
            "Worker gpu_0 destroyed.",
        ]

    @pytest.mark.asyncio
    async def test_a_running_worker_is_logged_as_stopped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        wm._destroy_worker = MagicMock()  # type: ignore[method-assign]
        worker = _worker(started=True)
        worker.status = WorkerStatus.RUNNING
        worker.stop = AsyncMock(return_value=True)

        with caplog.at_level(logging.INFO, logger="test.supervisor"):
            assert await wm._stop_and_destroy_worker(worker) is True

        worker.stop.assert_awaited_once_with()
        assert _info_messages(caplog) == [
            "Stopping worker gpu_0...",
            "Worker gpu_0 stopped.",
        ]
