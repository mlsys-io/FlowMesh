"""Latched per-run cancel and stop signals, and their callbacks."""

import threading
import time
from collections.abc import Callable

import pytest

from worker.executors.base_executor import RunControl, TaskCancelledError


class TestSignals:
    def test_signals_start_clear(self) -> None:
        control = RunControl("tsk-a")
        assert control.task_id == "tsk-a"
        assert not control.cancel_requested
        assert not control.stop_requested

    def test_cancel_and_stop_are_independent(self) -> None:
        cancelled = RunControl("tsk-a")
        cancelled.request_cancel()
        assert cancelled.cancel_requested
        assert not cancelled.stop_requested

        stopped = RunControl("tsk-b")
        stopped.request_stop()
        assert stopped.stop_requested
        assert not stopped.cancel_requested

    def test_raise_if_cancelled(self) -> None:
        control = RunControl("tsk-a")
        control.raise_if_cancelled("not raised")
        control.request_cancel()
        with pytest.raises(TaskCancelledError, match="gone"):
            control.raise_if_cancelled("gone")

    def test_wait_for_cancel_times_out_without_a_signal(self) -> None:
        control = RunControl("tsk-a")
        started = time.monotonic()
        assert control.wait_for_cancel(0.05) is False
        assert time.monotonic() - started >= 0.05

    def test_wait_for_cancel_returns_early_on_cancel(self) -> None:
        control = RunControl("tsk-a")
        threading.Timer(0.05, control.request_cancel).start()
        started = time.monotonic()
        assert control.wait_for_cancel(10.0) is True
        assert time.monotonic() - started < 5.0


class TestCallbacks:
    def test_callback_registered_before_the_signal_runs_on_signal(self) -> None:
        control = RunControl("tsk-a")
        calls: list[str] = []
        control.on_cancel(lambda: calls.append("cancel"))
        control.on_stop(lambda: calls.append("stop"))
        assert calls == []

        control.request_stop()
        assert calls == ["stop"]
        control.request_cancel()
        assert calls == ["stop", "cancel"]

    def test_callback_registered_after_the_signal_runs_immediately(self) -> None:
        control = RunControl("tsk-a")
        control.request_cancel()
        calls: list[threading.Thread] = []
        unregister = control.on_cancel(lambda: calls.append(threading.current_thread()))
        assert calls == [threading.current_thread()]
        unregister()

    def test_repeat_signal_fires_nothing(self) -> None:
        control = RunControl("tsk-a")
        calls: list[int] = []
        control.on_cancel(lambda: calls.append(1))
        control.request_cancel()
        control.request_cancel()
        assert calls == [1]

    def test_unregistered_callback_never_runs(self) -> None:
        control = RunControl("tsk-a")
        calls: list[int] = []
        unregister = control.on_cancel(lambda: calls.append(1))
        unregister()
        unregister()
        control.request_cancel()
        assert calls == []

    def test_unregister_after_the_callback_ran_is_a_no_op(self) -> None:
        control = RunControl("tsk-a")
        calls: list[int] = []
        unregister = control.on_stop(lambda: calls.append(1))
        control.request_stop()
        unregister()
        assert calls == [1]

    def test_raising_callback_does_not_stop_later_callbacks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        control = RunControl("tsk-a")
        calls: list[int] = []

        def boom() -> None:
            raise RuntimeError("boom")

        control.on_cancel(boom)
        control.on_cancel(lambda: calls.append(1))
        with caplog.at_level("WARNING"):
            control.request_cancel()

        assert calls == [1]
        assert any("tsk-a" in record.getMessage() for record in caplog.records)

    def test_raising_callback_registered_after_the_signal_is_swallowed(self) -> None:
        control = RunControl("tsk-a")
        control.request_stop()

        def boom() -> None:
            raise RuntimeError("boom")

        control.on_stop(boom)

    def test_concurrent_signals_fire_each_callback_once(self) -> None:
        control = RunControl("tsk-a")
        counts = [0] * 8
        lock = threading.Lock()

        def make(i: int) -> Callable[[], None]:
            def fn() -> None:
                with lock:
                    counts[i] += 1

            return fn

        for i in range(len(counts)):
            control.on_cancel(make(i))

        barrier = threading.Barrier(16)

        def signal() -> None:
            barrier.wait()
            control.request_cancel()

        threads = [threading.Thread(target=signal) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert counts == [1] * len(counts)
