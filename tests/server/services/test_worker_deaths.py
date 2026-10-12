"""Watchdog death marks shared with the scheduler."""

import logging
from typing import Any, cast
from unittest import mock

from server.services.watchdog import WorkerWatchdog
from server.services.worker_deaths import WorkerDeathMarks


def test_marks_track_dead_workers() -> None:
    deaths = WorkerDeathMarks()
    deaths.mark("wkr-1")
    assert deaths.is_dead("wkr-1")
    assert deaths.snapshot() == {"wkr-1"}
    deaths.clear("wkr-1")
    assert not deaths.is_dead("wkr-1")


def test_disabled_marks_never_report_dead() -> None:
    deaths = WorkerDeathMarks(enabled=False)
    deaths.mark("wkr-1")
    assert not deaths.is_dead("wkr-1")


def test_watchdog_records_deaths_in_the_shared_marks() -> None:
    deaths = WorkerDeathMarks()
    watchdog = WorkerWatchdog(
        cast(Any, mock.Mock()),
        cast(Any, mock.Mock()),
        cast(Any, mock.Mock()),
        cast(Any, mock.Mock()),
        logging.getLogger("worker-deaths"),
        enabled=True,
        check_interval=30,
        grace_seconds=0,
        death_marks=deaths,
    )
    watchdog._mark_dead("wkr-1")
    assert deaths.is_dead("wkr-1")
    watchdog.clear_dead_mark("wkr-1")
    assert not deaths.is_dead("wkr-1")
