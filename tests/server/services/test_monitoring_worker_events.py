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


def test_heartbeat_from_unregistered_worker_ignored() -> None:
    registry = MagicMock()
    registry.worker_is_registered.return_value = False
    monitor = _monitor(registry)
    monitor._handle_worker_event(_heartbeat("wkr-1"))
    registry.update_worker_hb.assert_not_called()


def test_heartbeat_from_registered_worker_updates() -> None:
    registry = MagicMock()
    registry.worker_is_registered.return_value = True
    monitor = _monitor(registry)
    monitor._handle_worker_event(_heartbeat("wkr-1"))
    registry.update_worker_hb.assert_called_once()
