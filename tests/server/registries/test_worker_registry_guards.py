"""Membership guards on the server WorkerRegistry write helpers.

A write to a worker that is no longer in ``WORKERS_SET_KEY`` (e.g. one the
watchdog reaped) must not recreate a partial record outside the set.
"""

from typing import Any, cast
from unittest.mock import MagicMock

from server.registries.worker import WorkerRegistry
from shared.schemas.worker import WorkerStatus


def _registry(member: bool) -> WorkerRegistry:
    rds: Any = MagicMock()
    rds.sync.sismember.return_value = member
    return WorkerRegistry(cast(Any, rds))


def test_unguarded_writers_skip_unregistered_worker() -> None:
    registry: Any = _registry(member=False)
    registry.update_worker_hb("wkr-1", "ts", 120)
    registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None)
    registry.update_worker_status("wkr-1", WorkerStatus.BUSY)
    registry._rds.sync.control_pipeline.assert_not_called()
    registry._rds.sync.hash_set.assert_not_called()
    registry._rds.sync.publish_telemetry.assert_not_called()


def test_unguarded_writers_still_write_registered_worker() -> None:
    registry: Any = _registry(member=True)
    registry.update_worker_hb("wkr-1", "ts", 120)
    registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None)
    registry.update_worker_status("wkr-1", WorkerStatus.BUSY)
    assert registry._rds.sync.control_pipeline.call_count == 2
    registry._rds.sync.hash_set.assert_called_once()
    registry._rds.sync.publish_telemetry.assert_called_once()
