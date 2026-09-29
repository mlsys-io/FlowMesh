"""TaskListener hands dispatches from other threads to per-worker streams without
occupying executor threads, and gives each worker id's queue a single owner."""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.task_listener import DispatchStream, TaskListener
from shared.schemas.command import TaskMessage

_LOGGER = logging.getLogger("test.task_listener")
_ANY_OBJECT: Any = None


def _run_in_thread(target: Callable[[], Any]) -> None:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=2)


class _RecordingExecutor(ThreadPoolExecutor):
    """Records submitted work and never runs it, so nothing can occupy a thread."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.submitted: list[Callable[..., Any]] = []

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        self.submitted.append(fn)
        return Future()


def test_dispatch_after_cancelled_get_events_uses_no_executor() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    executor = _RecordingExecutor()
    loop = asyncio.new_event_loop()
    loop.set_default_executor(executor)
    listener._loop = loop

    async def scenario() -> dict[str, Any] | None:
        dead = []
        for i in range(10):
            worker_id = f"wkr-dead-{i}"
            listener.add_worker(worker_id)
            stream = listener.attach_stream(worker_id)
            assert stream is not None
            dead.append(asyncio.ensure_future(stream.next()))
        await asyncio.sleep(0.01)
        for getter in dead:
            getter.cancel()
        await asyncio.gather(*dead, return_exceptions=True)

        listener.add_worker("wkr-live")
        live_stream = listener.attach_stream("wkr-live")
        assert live_stream is not None
        live = asyncio.ensure_future(live_stream.next())
        await asyncio.sleep(0)
        message = TaskMessage(worker_id="wkr-live", payload={"task_id": "tsk-1"})
        _run_in_thread(lambda: listener._handle_message(message.model_dump()))
        try:
            return await asyncio.wait_for(live, timeout=2)
        except TimeoutError:
            return None

    try:
        result = loop.run_until_complete(scenario())
    finally:
        loop.close()
    assert executor.submitted == []
    assert result == {"task_id": "tsk-1"}


def test_dispatch_relay_from_another_thread_arrives_in_order() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    accepted: list[bool] = []

    async def scenario() -> list[dict[str, Any] | None]:
        listener._loop = asyncio.get_running_loop()
        listener.add_worker("wkr-1")
        stream = listener.attach_stream("wkr-1")
        assert stream is not None

        def relay() -> None:
            for token in ("rly-1", "rly-2"):
                accepted.append(listener.dispatch_relay("wkr-1", token, "ep-1"))
            accepted.append(listener.dispatch_relay("wkr-unknown", "rly-3", "ep-1"))

        _run_in_thread(relay)
        return [await asyncio.wait_for(stream.next(), timeout=2) for _ in range(2)]

    events = asyncio.run(scenario())
    assert [event and event["relay_token"] for event in events] == ["rly-1", "rly-2"]
    assert accepted == [True, True, False]


def _attach(listener: TaskListener, worker_id: str) -> DispatchStream:
    stream = listener.attach_stream(worker_id)
    assert stream is not None
    return stream


async def _next(stream: DispatchStream) -> dict[str, Any] | None:
    return await asyncio.wait_for(stream.next(), timeout=2)


def test_remove_from_another_thread_drops_queued_frames_and_ends_the_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)

    async def scenario() -> list[dict[str, Any] | None]:
        listener._loop = asyncio.get_running_loop()
        listener.add_worker("wkr-1")
        stream = _attach(listener, "wkr-1")
        listener._deliver("wkr-1", {"task_id": "A"})

        _run_in_thread(lambda: listener.remove_worker("wkr-1"))
        await asyncio.sleep(0)
        results = [await _next(stream), await _next(stream)]

        message = TaskMessage(worker_id="wkr-1", payload={"task_id": "tsk-1"})
        _run_in_thread(lambda: listener._handle_message(message.model_dump()))
        _run_in_thread(lambda: listener.dispatch_relay("wkr-1", "rly-1", "ep-1"))
        await asyncio.sleep(0.01)
        return results

    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        results = asyncio.run(scenario())

    assert results == [None, None]
    assert "wkr-1" not in listener._qs
    assert listener.attach_stream("wkr-1") is None
    assert "Dropping 1 queued dispatch(es) for released worker: wkr-1" in caplog.text
    assert "Dropping dispatch for unregistered worker: wkr-1" in caplog.text


def test_new_stream_takes_over_pending_frames_in_order() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)

    async def scenario() -> tuple[Any, ...]:
        listener._loop = asyncio.get_running_loop()
        listener.add_worker("wkr-1")
        old = _attach(listener, "wkr-1")
        old_read = asyncio.ensure_future(old.next())
        await asyncio.sleep(0)

        # A wakes the old reader, but it hasn't resumed to take A off the queue
        # when the new stream attaches.
        listener._deliver("wkr-1", {"task_id": "A"})
        listener._deliver("wkr-1", {"task_id": "B"})
        new = _attach(listener, "wkr-1")

        old_result = await asyncio.wait_for(old_read, timeout=2)
        listener._deliver("wkr-1", {"task_id": "C"})
        new_results = [await _next(new) for _ in range(3)]
        old_after = await _next(old)
        return old_result, new_results, old_after

    old_result, new_results, old_after = asyncio.run(scenario())

    assert old_result is None
    assert new_results == [{"task_id": "A"}, {"task_id": "B"}, {"task_id": "C"}]
    assert old_after is None


def test_cancelled_woken_reader_leaves_its_frame_for_the_next_stream() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)

    async def scenario() -> dict[str, Any] | None:
        listener._loop = asyncio.get_running_loop()
        listener.add_worker("wkr-1")
        old_read = asyncio.ensure_future(_attach(listener, "wkr-1").next())
        await asyncio.sleep(0)

        listener._deliver("wkr-1", {"task_id": "A"})
        old_read.cancel()
        await asyncio.gather(old_read, return_exceptions=True)

        return await _next(_attach(listener, "wkr-1"))

    assert asyncio.run(scenario()) == {"task_id": "A"}


def test_attach_logs_only_when_superseding_an_attached_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    listener.add_worker("wkr-1")

    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        first = _attach(listener, "wkr-1")
        listener.detach_stream(first)
        second = _attach(listener, "wkr-1")
        assert "Superseding" not in caplog.text

        _attach(listener, "wkr-1")
        # A superseded stream finishing must not clear its successor's record.
        listener.detach_stream(second)

    assert caplog.text.count("Superseding task stream for worker wkr-1") == 1
    assert "wkr-1" in listener._attached


def test_attach_to_unknown_worker_returns_none() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    assert listener.attach_stream("wkr-unknown") is None


def test_remove_before_start_drops_queue_directly() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    listener.add_worker("wkr-1")

    listener.remove_worker("wkr-1")

    assert listener._qs == {}


def test_replaced_bindings_keep_one_queue_per_token() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    registry = WorkerRegistry(on_worker_id_released=listener.remove_worker)
    token: Any = "tok-1"

    async def scenario() -> None:
        listener._loop = asyncio.get_running_loop()
        for i in range(50):
            worker_id = f"wkr-{i}"
            registry.set_worker_id(token, worker_id)
            listener.add_worker(worker_id)
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert list(listener._qs) == ["wkr-49"]
