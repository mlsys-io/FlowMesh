"""EventMonitor worker-event handling: heartbeat gating on registration."""

import logging
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor
from shared.schemas.event import WorkerEvent


def _monitor(worker_registry: MagicMock) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test-monitor"),
        runtime=MagicMock(),
        dispatcher=MagicMock(),
        worker_registry=worker_registry,
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
    )


def _heartbeat(worker_id: str) -> WorkerEvent:
    return WorkerEvent(type="HEARTBEAT", worker_id=worker_id, payload={"ttl_sec": 120})


def _status(worker_id: str) -> WorkerEvent:
    return WorkerEvent(type="STATUS", worker_id=worker_id, payload={})


def test_heartbeat_from_unregistered_worker_ignored() -> None:
    registry = MagicMock()
    registry.update_worker_hb.return_value = False
    monitor = _monitor(registry)
    monitor._handle_worker_event(_heartbeat("wkr-1"))
    registry.update_worker_hb.assert_called_once()


def test_heartbeat_from_registered_worker_updates() -> None:
    registry = MagicMock()
    registry.update_worker_hb.return_value = True
    monitor = _monitor(registry)
    monitor._handle_worker_event(_heartbeat("wkr-1"))
    registry.update_worker_hb.assert_called_once()


def test_status_from_unregistered_worker_ignored() -> None:
    registry = MagicMock()
    registry.set_worker_status.return_value = False
    monitor = _monitor(registry)
    monitor._handle_worker_event(_status("wkr-1"))
    registry.set_worker_status.assert_called_once()


def test_status_from_registered_worker_updates() -> None:
    registry = MagicMock()
    registry.set_worker_status.return_value = True
    monitor = _monitor(registry)
    monitor._handle_worker_event(_status("wkr-1"))
    registry.set_worker_status.assert_called_once()


class TestHeartbeatCarriesGpuOccupancy:
    def _occupancy_heartbeat(self, worker_id: str) -> WorkerEvent:
        return WorkerEvent(
            type="HEARTBEAT",
            worker_id=worker_id,
            payload={"ttl_sec": 120},
            metrics={
                "gpu_occupancy": {"GPU-held": {"unavailable": True, "free_bytes": 8}}
            },
        )

    def test_occupancy_is_recorded(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        _monitor(registry)._handle_worker_event(self._occupancy_heartbeat("wkr-1"))
        registry.record_gpu_occupancy.assert_called_once_with(
            "wkr-1", {"GPU-held": {"unavailable": True, "free_bytes": 8}}
        )

    def test_unknown_worker_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = False
        _monitor(registry)._handle_worker_event(self._occupancy_heartbeat("wkr-1"))
        registry.record_gpu_occupancy.assert_not_called()

    def test_a_recording_failure_does_not_escape(self) -> None:
        # Scheduling advice must never cost the worker its liveness update.
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        registry.record_gpu_occupancy.side_effect = RuntimeError("redis down")
        _monitor(registry)._handle_worker_event(self._occupancy_heartbeat("wkr-1"))
        registry.update_worker_hb.assert_called_once()

    def test_heartbeat_without_occupancy_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        _monitor(registry)._handle_worker_event(_heartbeat("wkr-1"))
        registry.record_gpu_occupancy.assert_not_called()
