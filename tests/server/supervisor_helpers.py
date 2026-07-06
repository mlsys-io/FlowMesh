"""Shared stubs for the supervisor lifecycle / re-register tests."""

import logging

from server.registries.node import NodeRegistry
from server.supervisor.services.lifecycle import Lifecycle

_LOGGER = logging.getLogger("test.supervisor")


class StubRegistry(NodeRegistry):
    """NodeRegistry whose ``node_exists`` returns a fixed value and records the
    ids it was queried with."""

    def __init__(self, exists: bool) -> None:
        self._exists = exists
        self.exists_calls: list[str] = []

    def node_exists(self, node_id: str) -> bool:
        self.exists_calls.append(node_id)
        return self._exists


class StubLifecycle(Lifecycle):
    """Lifecycle with registration stubbed: re-register mints ``nde-2`` and each
    published event type is recorded instead of sent."""

    def __init__(self, node_registry: NodeRegistry, node_id: str) -> None:
        self._node_registry = node_registry
        self._node_id = node_id
        self.logger = _LOGGER
        self._unregister_published = True
        self._on_reregister = None
        self.published_events: list[str] = []

    def _register(self) -> str:
        return "nde-2"

    def _publish_event(self, event_type: str, **extra: object) -> None:
        self.published_events.append(event_type)
