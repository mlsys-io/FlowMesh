"""Process-local worker death marks on the worker registry."""

from typing import Any, cast
from unittest import mock

from server.registries.worker import WorkerRegistry


def test_death_marks_track_marked_workers() -> None:
    registry = WorkerRegistry(cast(Any, mock.Mock()))
    registry.mark_dead("wkr-1")
    assert registry.is_marked_dead("wkr-1")
    assert registry.dead_marks() == {"wkr-1"}
    registry.clear_dead_mark("wkr-1")
    assert not registry.is_marked_dead("wkr-1")
    assert registry.dead_marks() == set()


def test_new_registry_has_no_death_marks() -> None:
    assert not WorkerRegistry(cast(Any, mock.Mock())).is_marked_dead("wkr-1")
