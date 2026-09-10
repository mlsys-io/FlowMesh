"""WorkerRegistry behaviour and thread-safety under concurrent access."""

import threading
from typing import cast

import pytest

from server.supervisor.adapters.base import WorkerAdapter, WorkerTokenType
from server.supervisor.registry import WorkerRegistry


class _FakeAdapter:
    def __init__(self, token: str, name: str) -> None:
        self.token = cast(WorkerTokenType, token)
        self.name = name


def _adapter(token: str, name: str) -> WorkerAdapter:
    return cast(WorkerAdapter, _FakeAdapter(token, name))


def test_add_get_pop_roundtrip() -> None:
    registry = WorkerRegistry()
    worker = _adapter("tok-1", "worker-1")

    registry.add(worker)
    assert registry.try_get(cast(WorkerTokenType, "tok-1")) is worker
    assert registry.try_get_by_name("worker-1") is worker
    assert registry.all_workers() == [worker]

    registry.set_worker_id(cast(WorkerTokenType, "tok-1"), "wrk-1")
    assert registry.get_worker_id(cast(WorkerTokenType, "tok-1")) == "wrk-1"

    popped = registry.try_pop(cast(WorkerTokenType, "tok-1"))
    assert popped is worker
    assert registry.all_workers() == []
    assert registry.try_get(cast(WorkerTokenType, "tok-1")) is None
    assert registry.get_worker_id(cast(WorkerTokenType, "tok-1")) is None


def test_add_rejects_duplicate_token_and_name() -> None:
    registry = WorkerRegistry()
    registry.add(_adapter("tok-1", "worker-1"))

    with pytest.raises(ValueError, match="token"):
        registry.add(_adapter("tok-1", "worker-2"))
    with pytest.raises(ValueError, match="name"):
        registry.add(_adapter("tok-2", "worker-1"))


def test_concurrent_mutation_and_snapshot_do_not_crash() -> None:
    """A mutating loop and an all_workers() reader must not race into a
    'dictionary changed size during iteration' error."""
    registry = WorkerRegistry()
    iterations = 20_000
    errors: list[BaseException] = []
    start = threading.Barrier(2)
    done = threading.Event()

    def mutate() -> None:
        start.wait()
        try:
            for i in range(iterations):
                token = cast(WorkerTokenType, f"tok-{i}")
                registry.add(_adapter(token, f"worker-{i}"))
                registry.try_pop(token)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def snapshot() -> None:
        start.wait()
        try:
            while not done.is_set():
                for worker in registry.all_workers():
                    _ = worker.name
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=mutate),
        threading.Thread(target=snapshot),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert registry.all_workers() == []
