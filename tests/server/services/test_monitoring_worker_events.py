"""EventMonitor worker-event handling: heartbeat gating on registration."""

import json
import logging
from unittest.mock import MagicMock

from server.registries.worker import WorkerRegistry
from server.services.monitoring import EventMonitor
from shared.schemas.event import WorkerEvent, parse_event
from shared.schemas.worker import WorkerStatus


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


class TestHeartbeatCarriesGpuAvailability:
    def _availability_heartbeat(self, worker_id: str) -> WorkerEvent:
        return WorkerEvent(
            type="HEARTBEAT",
            worker_id=worker_id,
            payload={"ttl_sec": 120},
            metrics={
                "gpu_availability": {"GPU-held": {"available": False, "free_bytes": 8}}
            },
        )

    def test_availability_is_recorded(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_called_once_with(
            "wkr-1", {"GPU-held": {"available": False, "free_bytes": 8}}
        )

    def test_unknown_worker_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = False
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_not_called()

    def test_a_recording_failure_does_not_escape(self) -> None:
        # Scheduling advice must never cost the worker its liveness update.
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        registry.record_gpu_availability.side_effect = RuntimeError("redis down")
        _monitor(registry)._handle_worker_event(self._availability_heartbeat("wkr-1"))
        registry.update_worker_hb.assert_called_once()

    def test_heartbeat_without_availability_records_nothing(self) -> None:
        registry = MagicMock()
        registry.update_worker_hb.return_value = True
        _monitor(registry)._handle_worker_event(_heartbeat("wkr-1"))
        registry.record_gpu_availability.assert_not_called()


class TestServerOriginStatusEvents:
    """A status event the server published is not written back to the registry."""

    def test_server_origin_status_is_not_reapplied(self) -> None:
        registry = MagicMock()
        monitor = _monitor(registry)
        event = WorkerEvent(
            type="STATUS",
            worker_id="wkr-1",
            status=WorkerStatus.BUSY,
            origin="server",
        )
        monitor._handle_worker_event(event)
        registry.set_worker_status.assert_not_called()

    def test_worker_origin_status_is_still_applied(self) -> None:
        registry = MagicMock()
        registry.set_worker_status.return_value = True
        monitor = _monitor(registry)
        event = WorkerEvent(
            type="STATUS",
            worker_id="wkr-1",
            status=WorkerStatus.IDLE,
            origin="worker",
        )
        monitor._handle_worker_event(event)
        registry.set_worker_status.assert_called_once()

    def test_registry_status_updates_round_trip_as_server_origin(self) -> None:
        rds = MagicMock()
        WorkerRegistry(rds).update_worker_status("wkr-1", WorkerStatus.BUSY)
        channel, raw = rds.sync.publish_telemetry.call_args.args
        event = parse_event(json.loads(raw))
        assert isinstance(event, WorkerEvent)
        assert event.origin == "server"

        registry = MagicMock()
        _monitor(registry)._handle_worker_event(event)
        registry.set_worker_status.assert_not_called()
