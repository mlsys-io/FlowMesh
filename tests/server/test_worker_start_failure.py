"""Tests for how the worker manager reports a worker that fails to start."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.supervisor.manager import WorkerManager
from server.supervisor.schemas import WorkerStatus


def _worker_manager() -> tuple[WorkerManager, MagicMock]:
    """A started manager plus the registry mock it was given."""
    registry = MagicMock()
    wm = object.__new__(WorkerManager)
    wm.config_path = "/dev/null"
    wm.logger = logging.getLogger("test-wm")
    wm._registry = registry
    wm._is_started = True
    wm._default_worker_config = {}
    wm._capacity_change_callback = None
    return wm, registry


def _worker(name: str = "gpu_0", *, started: bool) -> MagicMock:
    worker = MagicMock()
    worker.alias = name
    worker.token = f"{name}.token"
    worker.status = WorkerStatus.STOPPED
    worker.start = AsyncMock(return_value=started)
    return worker


class TestStartWorkerFailure:
    @pytest.mark.asyncio
    async def test_failed_start_is_logged_as_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm, registry = _worker_manager()
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=False)

        with caplog.at_level(logging.ERROR, logger="test-wm"):
            result = await wm._start_worker(worker)

        assert result is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "gpu_0" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_failed_start_still_discards_the_worker(self) -> None:
        wm, registry = _worker_manager()
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=False)

        assert await wm._start_worker(worker) is False
        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)

    @pytest.mark.asyncio
    async def test_successful_start_logs_nothing_and_keeps_the_worker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm, registry = _worker_manager()
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=True)

        with caplog.at_level(logging.ERROR, logger="test-wm"):
            result = await wm._start_worker(worker)

        assert result is True
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        wm._stop_and_destroy_worker.assert_not_awaited()
        registry.try_pop.assert_not_called()
