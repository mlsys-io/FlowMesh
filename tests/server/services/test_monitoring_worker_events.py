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
