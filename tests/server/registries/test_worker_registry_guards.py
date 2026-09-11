"""Membership guards on the server WorkerRegistry write helpers.

A write to a worker that is no longer in ``WORKERS_SET_KEY`` (e.g. one the
watchdog reaped) must not recreate a partial record outside the set. The guard
and the write are one atomic Redis script, so the registry never reads
membership separately.
"""

from typing import Any, cast
from unittest.mock import MagicMock

from server.registries.worker import WorkerRegistry
from shared.schemas.worker import WorkerStatus


def _registry(wrote: int) -> WorkerRegistry:
    rds: Any = MagicMock()
    rds.sync.eval.return_value = wrote
    return WorkerRegistry(cast(Any, rds))


def test_writers_report_a_skipped_unregistered_worker() -> None:
    registry: Any = _registry(wrote=0)
    assert registry.update_worker_hb("wkr-1", "ts", 120) is False
    assert registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None) is False
    assert registry.update_worker_status("wkr-1", WorkerStatus.BUSY) is False
    # A skipped status write must not announce a status it never stored.
    registry._rds.sync.publish_telemetry.assert_not_called()


def test_writers_report_a_registered_worker() -> None:
    registry: Any = _registry(wrote=1)
    assert registry.update_worker_hb("wkr-1", "ts", 120) is True
    assert registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None) is True
    assert registry.update_worker_status("wkr-1", WorkerStatus.BUSY) is True
    assert registry._rds.sync.eval.call_count == 3
    registry._rds.sync.publish_telemetry.assert_called_once()


def test_writes_are_a_single_atomic_call() -> None:
    registry: Any = _registry(wrote=1)
    registry.update_worker_hb("wkr-1", "ts", 120)
    # No separate membership read, and no pipeline that could interleave a reap.
    registry._rds.sync.sismember.assert_not_called()
    registry._rds.sync.control_pipeline.assert_not_called()


def test_status_extras_are_prefixed_in_the_script_arguments() -> None:
    registry: Any = _registry(wrote=1)
    registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", {"gpu": 2})
    args = registry._rds.sync.eval.call_args.args
    # numkeys, the two keys, then worker_id followed by field/value pairs.
    assert args[1] == 2
    assert args[4] == "wkr-1"
    assert "extra_gpu" in args
    assert args[args.index("extra_gpu") + 1] == "2"
