"""Per-device GPU occupancy: what the worker observes and what it reports."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.schemas.worker import WorkerStatus
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EchoSpecStrict, InferenceSpecStrict
from shared.tasks.task_type import TaskType
from worker.gpu_occupancy import (
    MIB,
    DeviceReading,
    DeviceState,
    GpuGateConfig,
    GpuOccupancyMonitor,
    NvmlDeviceProbe,
    decide,
)
from worker.lifecycle import Lifecycle
from worker.runner import Runner

THRESH = 1024
GPU_A = "GPU-aaaa"
GPU_B = "GPU-bbbb"


def _reading(used_mib: float, free_bytes: int = 0) -> DeviceReading:
    return DeviceReading(used_mib=used_mib, free_bytes=free_bytes)


class TestDecide:
    def test_needs_consecutive_observations_to_enter(self) -> None:
        state = decide(40_000, THRESH, 2, DeviceState())
        assert state.unavailable is False and state.streak == 1
        assert decide(40_000, THRESH, 2, state).unavailable is True

    def test_needs_consecutive_observations_to_leave(self) -> None:
        held = DeviceState(unavailable=True)
        state = decide(5, THRESH, 2, held)
        assert state.unavailable is True and state.streak == 1
        assert decide(5, THRESH, 2, state).unavailable is False

    def test_single_spike_does_not_flip(self) -> None:
        state = decide(40_000, THRESH, 2, DeviceState())
        assert decide(5, THRESH, 2, state).streak == 0

    def test_unreadable_device_keeps_state(self) -> None:
        held = DeviceState(unavailable=True)
        assert decide(None, THRESH, 2, held).unavailable is True

    def test_at_threshold_is_not_occupied(self) -> None:
        assert decide(THRESH, THRESH, 1, DeviceState()).unavailable is False

    def test_consecutive_one_flips_immediately(self) -> None:
        assert decide(40_000, THRESH, 1, DeviceState()).unavailable is True


class TestNvmlDeviceProbe:
    def _fake_nvml(self, devices: dict[int, tuple[str, str, int, int]]) -> Any:
        class FakeNvml:
            NVMLError = RuntimeError

            @staticmethod
            def nvmlInit() -> None:
                return None

            @staticmethod
            def nvmlDeviceGetCount() -> int:
                return len(devices)

            @staticmethod
            def nvmlDeviceGetHandleByIndex(index: int) -> int:
                return index

            @staticmethod
            def nvmlDeviceGetName(handle: int) -> str:
                return devices[handle][0]

            @staticmethod
            def nvmlDeviceGetUUID(handle: int) -> str:
                return devices[handle][1]

            @staticmethod
            def nvmlDeviceGetMemoryInfo(handle: int) -> Any:
                _, _, used, free = devices[handle]
                return SimpleNamespace(used=used, free=free)

        return FakeNvml

    def test_keys_by_uuid_not_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Index is only meaningful relative to CUDA_VISIBLE_DEVICES.
        monkeypatch.setattr(
            "worker.gpu_occupancy.pynvml",
            self._fake_nvml({0: ("dedicated", GPU_A, 40_000 * MIB, 8 * MIB)}),
        )
        assert NvmlDeviceProbe()() == {GPU_A: _reading(40_000.0, free_bytes=8 * MIB)}

    def test_unified_devices_are_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Their "used" figure is system RAM, not a card another tenant holds --
        # so they report nothing rather than reporting free.
        monkeypatch.setattr(
            "worker.gpu_occupancy.pynvml",
            self._fake_nvml(
                {
                    0: ("unified", GPU_A, 40_000 * MIB, 0),
                    1: ("dedicated", GPU_B, 0, 48 * 1024 * MIB),
                }
            ),
        )
        readings = NvmlDeviceProbe(lambda index, _name: index == 0)()
        assert set(readings) == {GPU_B}

    def test_nvml_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            NVMLError = RuntimeError

            @staticmethod
            def nvmlInit() -> None:
                raise RuntimeError("driver wedged")

        monkeypatch.setattr("worker.gpu_occupancy.pynvml", Broken)
        assert NvmlDeviceProbe()() == {}


class TestMonitor:
    def _monitor(
        self, batches: list[dict[str, DeviceReading]], **cfg: Any
    ) -> GpuOccupancyMonitor:
        config = GpuGateConfig(
            enabled=cfg.get("enabled", True),
            threshold_mib=THRESH,
            consecutive=cfg.get("consecutive", 2),
            grace_sec=0.0,
        )
        it = iter(batches)
        return GpuOccupancyMonitor(config, lambda: next(it))

    def test_devices_are_tracked_independently(self) -> None:
        held = {GPU_A: _reading(40_000), GPU_B: _reading(5)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].unavailable is True
        assert snapshot[GPU_B].unavailable is False

    def test_suppressed_observation_latches(self) -> None:
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].unavailable is True
        # Nothing is probed while suppressed, so the batch list is not consumed.
        monitor.observe(False)
        assert monitor.snapshot()[GPU_A].unavailable is True
        assert monitor.measured is False

    def test_total_probe_failure_clears_rather_than_latching(self) -> None:
        # Only a fresh clear reading releases a latch, and a broken probe never
        # produces one -- latching here would exclude the worker forever.
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held, {}])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].unavailable is True
        monitor.observe(True)
        assert monitor.snapshot() == {}
        assert monitor.measured is False

    def test_partial_failure_leaves_unread_devices_latched(self) -> None:
        both = {GPU_A: _reading(40_000), GPU_B: _reading(40_000)}
        monitor = self._monitor([both, both, {GPU_B: _reading(5)}])
        monitor.observe(True)
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].unavailable is True, "unread device keeps its state"
        assert snapshot[GPU_B].unavailable is True, "one clear reading is not enough"

    def test_free_bytes_ride_along(self) -> None:
        monitor = self._monitor([{GPU_A: _reading(5, free_bytes=1234)}])
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].free_bytes == 1234

    def test_disabled_monitor_never_probes(self) -> None:
        def explode() -> dict[str, DeviceReading]:
            pytest.fail("probed while disabled")

        monitor = GpuOccupancyMonitor(GpuGateConfig(enabled=False), explode)
        monitor.observe(True)
        assert monitor.snapshot() == {}


class FakeClient:
    def __init__(self) -> None:
        self.statuses: list[tuple[WorkerStatus, dict[str, Any] | None]] = []

    def set_status(self, status: WorkerStatus, extra: dict | None = None) -> None:
        self.statuses.append((status, extra))


def _lifecycle(
    tmp_path: Path, batches: list[dict[str, DeviceReading]], grace: float = 0.0
) -> tuple[Lifecycle, GpuOccupancyMonitor, FakeClient]:
    client = FakeClient()
    it = iter(batches)
    monitor = GpuOccupancyMonitor(
        GpuGateConfig(
            enabled=True, threshold_mib=THRESH, consecutive=1, grace_sec=grace
        ),
        lambda: next(it),
    )
    lc = Lifecycle(
        client,  # type: ignore[arg-type]
        30,
        120,
        tmp_path / "hb",
        cost_per_hour=0.0,
        gpu_monitor=monitor,
    )
    lc._reported = WorkerStatus.IDLE  # as after start()
    lc.set_gpu_executor_probe(lambda: False)
    return lc, monitor, client


class TestLifecycleIntegration:
    def test_occupancy_is_reported_without_touching_status(
        self, tmp_path: Path
    ) -> None:
        # The whole point of this change: the worker stays IDLE and keeps taking
        # CPU work while its GPU is reported unavailable.
        lc, _, client = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert client.statuses == []
        assert lc._metrics()["gpu_occupancy"][GPU_A]["unavailable"] is True

    def test_a_warm_gpu_executor_suppresses_the_reading(self, tmp_path: Path) -> None:
        # Regression guard: reading our own resident model and calling it foreign
        # is the bug 49058d7 fixed, and it must not come back by another route.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc.set_gpu_executor_probe(lambda: True)
        lc._observe_gpu()
        assert monitor.snapshot() == {}
        assert monitor.measured is False

    def test_an_active_task_suppresses_the_reading(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc.set_busy("tsk-1")
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_no_registered_probe_is_unmeasurable(self, tmp_path: Path) -> None:
        # main.py registers the probe only after Runner exists; until then we
        # cannot know whether an executor is warm, so we must not measure.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._gpu_executor_probe = None
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_grace_window_suppresses_after_a_task(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}], grace=600)
        lc.set_busy("tsk-1")
        lc.set_idle("tsk-1")
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_a_finished_task_no_longer_wipes_occupancy(self, tmp_path: Path) -> None:
        # set_idle used to clear the gate, which is why admission once had to run
        # before set_busy. The latch now survives a task boundary.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert monitor.snapshot()[GPU_A].unavailable is True
        lc.set_busy("tsk-1")
        lc.set_idle("tsk-1")
        assert monitor.snapshot()[GPU_A].unavailable is True

    def test_live_occupancy_hides_a_stale_latch(self, tmp_path: Path) -> None:
        # Refusing a task on a latch we cannot currently confirm would fail it
        # terminally; the advisory snapshot keeps it, the live view does not.
        lc, _, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert lc.live_gpu_occupancy()[GPU_A].unavailable is True
        lc.set_gpu_executor_probe(lambda: True)
        lc._observe_gpu()
        assert lc.live_gpu_occupancy() == {}
        assert lc._metrics()["gpu_occupancy"][GPU_A]["unavailable"] is True


class TestWarmExecutorGpuFlag:
    """The flag behind ``has_active_gpu_executor``.

    Reading GPU-ness off the executor class does not work: the default config
    wraps most executors in ``MPExecutor``, whose class carries no such
    attribute, and a transformers executor's device depends on the spec it ran.
    So the flag is recorded from each task's own spec.
    """

    def _runner(self) -> Runner:
        return Runner(
            lifecycle=MagicMock(),
            task_stream=[],
            results_dir=Path("/tmp/unused"),
            hardware=MagicMock(),
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

    def _spec(self, *, gpu: bool) -> Any:
        if gpu:
            return InferenceSpecStrict(
                taskType=TaskType.INFERENCE,
                data={"type": "list", "items": ["hi"]},
                model=ModelConfig(source=ModelSource(identifier="org/m")),
            )
        return EchoSpecStrict(taskType=TaskType.ECHO)

    def test_no_executor_means_no_gpu_held(self) -> None:
        runner = self._runner()
        runner._note_gpu_usage(self._spec(gpu=True))
        assert runner.has_active_gpu_executor() is False

    def test_a_gpu_task_marks_the_warm_executor(self) -> None:
        runner = self._runner()
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        assert runner.has_active_gpu_executor() is True

    def test_a_cpu_task_alone_does_not(self) -> None:
        runner = self._runner()
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=False))
        assert runner.has_active_gpu_executor() is False

    def test_a_later_cpu_task_does_not_clear_an_earlier_gpu_task(self) -> None:
        # The reuse hole: a CPU task following a GPU task on the same warm
        # executor must not make us forget the VRAM the GPU task allocated.
        runner = self._runner()
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        runner._note_gpu_usage(self._spec(gpu=False))
        assert runner.has_active_gpu_executor() is True

    def test_teardown_clears_the_flag(self) -> None:
        runner = self._runner()
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        runner._cleanup_active_executor()
        assert runner._active_executor_used_gpu is False
        assert runner.has_active_gpu_executor() is False
