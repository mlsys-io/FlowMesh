"""WorkerWatchdog stale-worker reaper tests.

Drive ``_scan`` directly with an explicit clock instead of sleeping, so the
death-declaration and reap passes can be stepped deterministically.
"""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

from server.services.watchdog import WorkerWatchdog, _WatchdogState


def _watchdog(**kwargs) -> Any:
    defaults: dict[str, Any] = dict(
        redis_client=MagicMock(),
        worker_registry=MagicMock(),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        logger=logging.getLogger("test-watchdog"),
        enabled=True,
        check_interval=30,
        grace_seconds=60,
        rehydration_grace_seconds=0,
        reap_enabled=True,
        reap_grace_seconds=900,
    )
    defaults.update(kwargs)
    return WorkerWatchdog(**cast(Any, defaults))


def _stale_registry(worker_ids: set[str]) -> Any:
    registry: Any = MagicMock()
    registry.get_worker_ids.return_value = worker_ids
    registry.is_worker_stale.return_value = True
    return registry


def test_reaps_after_grace() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    # First pass only seeds stale_since; no declaration yet.
    wd._runtime.recover_tasks_for_worker.assert_not_called()
    registry.unregister_workers.assert_not_called()

    wd._scan({"wkr-1"}, state, 60.0)
    # Declaration at t=60: recovery runs, purge does not.
    wd._runtime.recover_tasks_for_worker.assert_called_once_with("wkr-1")
    registry.unregister_workers.assert_not_called()

    wd._scan({"wkr-1"}, state, 60.0 + 900.0)
    registry.unregister_workers.assert_called_once_with("wkr-1")
    # The synthetic UNREGISTER is published on the worker event channel.
    wd._redis.publish_telemetry.assert_called_once()
    channel = wd._redis.publish_telemetry.call_args.args[0]
    assert channel == "workers:events"


def test_no_reap_before_grace() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    wd._scan({"wkr-1"}, state, 60.0 + 900.0 - 1.0)
    registry.unregister_workers.assert_not_called()


def test_declared_dead_worker_reaps_on_schedule() -> None:
    """Regression: the reap must land at declaration + reap_grace, not
    declaration + death_grace, even when reap_grace < death_grace."""
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=10
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    wd._scan({"wkr-1"}, state, 60.0 + 10.0)
    registry.unregister_workers.assert_called_once_with("wkr-1")


def test_never_reaps_live_worker() -> None:
    registry: Any = MagicMock()
    registry.get_worker_ids.return_value = {"wkr-1"}
    registry.is_worker_stale.return_value = False
    wd: Any = _watchdog(worker_registry=registry)
    state = _WatchdogState()

    for t in (0.0, 30.0, 60.0, 1000.0):
        wd._scan({"wkr-1"}, state, t)
    wd._runtime.recover_tasks_for_worker.assert_not_called()
    registry.unregister_workers.assert_not_called()


def test_no_reap_after_worker_returns() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    # Heartbeat resumes: the worker is no longer stale.
    registry.is_worker_stale.return_value = False
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)
    registry.unregister_workers.assert_not_called()
    assert not wd.is_marked_dead("wkr-1")


def test_reaps_old_id_only_after_reenrollment() -> None:
    registry: Any = _stale_registry({"wkr-1", "wkr-2"})
    registry.is_worker_stale.side_effect = lambda wid: wid == "wkr-1"
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1", "wkr-2"}, state, 0.0)
    wd._scan({"wkr-1", "wkr-2"}, state, 60.0)  # only wkr-1 declared
    wd._scan({"wkr-1", "wkr-2"}, state, 60.0 + 900.0)
    registry.unregister_workers.assert_called_once_with("wkr-1")


def test_reap_disabled() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry,
        grace_seconds=60,
        reap_grace_seconds=900,
        reap_enabled=False,
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)
    wd._runtime.recover_tasks_for_worker.assert_called_once_with("wkr-1")
    registry.unregister_workers.assert_not_called()


def test_reap_failure_is_retried() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    registry.unregister_workers.side_effect = RuntimeError("redis down")
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)  # reap attempt fails, no raise
    assert registry.unregister_workers.call_count == 1
    assert "wkr-1" in state.dead_since  # kept for retry

    registry.unregister_workers.side_effect = None
    wd._scan({"wkr-1"}, state, 60.0 + 900.0 + 30.0)
    assert registry.unregister_workers.call_count == 2


def test_reaped_worker_is_deleted_once_more_next_pass() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)  # reaped
    assert registry.unregister_workers.call_count == 1
    assert "wkr-1" in state.reaped

    # Next pass re-deletes the reaped id exactly once, then drops it.
    wd._scan(set(), state, 60.0 + 900.0 + 30.0)
    assert registry.unregister_workers.call_count == 2
    assert "wkr-1" not in state.reaped


def test_reap_publish_failure_is_retried() -> None:
    registry = _stale_registry({"wkr-1"})
    wd = _watchdog(worker_registry=registry, grace_seconds=60, reap_grace_seconds=900)
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared
    # Publish fails on the reap pass: the id stays in reaped for retry.
    wd._redis.publish_telemetry.side_effect = RuntimeError("redis down")
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)
    assert "wkr-1" in state.reaped

    # Next pass re-deletes and re-publishes; success drops the id.
    wd._redis.publish_telemetry.side_effect = None
    wd._scan(set(), state, 60.0 + 900.0 + 30.0)
    assert "wkr-1" not in state.reaped
    assert wd._redis.publish_telemetry.call_count == 2


def test_dead_mark_survives_reap() -> None:
    registry: Any = _stale_registry({"wkr-1"})
    wd: Any = _watchdog(
        worker_registry=registry, grace_seconds=60, reap_grace_seconds=900
    )
    state = _WatchdogState()

    wd._scan({"wkr-1"}, state, 0.0)
    wd._scan({"wkr-1"}, state, 60.0)  # declared -> marked dead
    assert wd.is_marked_dead("wkr-1")
    wd._scan({"wkr-1"}, state, 60.0 + 900.0)  # reaped
    # The monitor, not the reaper, clears the mark.
    assert wd.is_marked_dead("wkr-1")
