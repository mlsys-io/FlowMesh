"""Per-device GPU availability: what the worker observes and what it reports."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.schemas.worker import WorkerStatus
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EchoSpecStrict, InferenceSpecStrict
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from worker.executors.base_executor import ExecutionError
from worker.gpu_availability import (
    MIB,
    DeviceAvailability,
    DeviceReading,
    DeviceState,
    GpuAvailabilityMonitor,
    GpuGateConfig,
    NvmlDeviceProbe,
    decide_availability,
)
from worker.lifecycle import Lifecycle
from worker.runner import Runner

THRESH = 1024
GPU_A = "GPU-aaaa"
GPU_B = "GPU-bbbb"


def _reading(used_mib: float, free_bytes: int = 0) -> DeviceReading:
    return DeviceReading(used_mib=used_mib, free_bytes=free_bytes)


class TestDecide:
    def _decide(
        self, used_mib: float, previous: DeviceState, consecutive: int = 2
    ) -> DeviceState:
        return decide_availability(_reading(used_mib), THRESH, consecutive, previous)

    def test_needs_consecutive_observations_to_enter(self) -> None:
        state = self._decide(40_000, DeviceState())
        assert state.availability.available is True and state.streak == 1
        assert self._decide(40_000, state).availability.available is False

    def test_needs_consecutive_observations_to_leave(self) -> None:
        held = DeviceState(DeviceAvailability(available=False))
        state = self._decide(5, held)
        assert state.availability.available is False and state.streak == 1
        assert self._decide(5, state).availability.available is True

    def test_single_spike_does_not_flip(self) -> None:
        state = self._decide(40_000, DeviceState())
        assert self._decide(5, state).streak == 0

    def test_at_threshold_is_still_available(self) -> None:
        assert self._decide(THRESH, DeviceState()).availability.available is True

    def test_consecutive_one_flips_immediately(self) -> None:
        state = self._decide(40_000, DeviceState(), consecutive=1)
        assert state.availability.available is False

    def test_free_bytes_always_come_from_the_latest_reading(self) -> None:
        # Even while latched mid-streak, the reported free figure is the fresh one.
        held = DeviceState(DeviceAvailability(available=False, free_bytes=1))
        state = decide_availability(_reading(5, free_bytes=999), THRESH, 2, held)
        assert state.availability == DeviceAvailability(available=False, free_bytes=999)


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
            "worker.gpu_availability.pynvml",
            self._fake_nvml({0: ("dedicated", GPU_A, 40_000 * MIB, 8 * MIB)}),
        )
        assert NvmlDeviceProbe()() == {GPU_A: _reading(40_000.0, free_bytes=8 * MIB)}

    def test_unified_devices_are_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Their "used" figure is system RAM, not a card another tenant holds --
        # so they report nothing rather than reporting free.
        monkeypatch.setattr(
            "worker.gpu_availability.pynvml",
            self._fake_nvml(
                {
                    0: ("unified", GPU_A, 40_000 * MIB, 0),
                    1: ("dedicated", GPU_B, 0, 48 * 1024 * MIB),
                }
            ),
        )
        readings = NvmlDeviceProbe(lambda index, _name: index == 0)()
        assert set(readings) == {GPU_B}

    def test_reads_only_the_workers_own_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NVML also lists GPUs CUDA_VISIBLE_DEVICES hides from the worker; those
        # belong to someone else and are not the worker's to report.
        monkeypatch.setattr(
            "worker.gpu_availability.pynvml",
            self._fake_nvml(
                {
                    0: ("dedicated", GPU_A, 40_000 * MIB, 8 * MIB),
                    1: ("dedicated", GPU_B, 0, 48 * 1024 * MIB),
                }
            ),
        )
        seen: list[int] = []

        def is_unified(ordinal: int, _name: str) -> bool:
            seen.append(ordinal)
            return False

        readings = NvmlDeviceProbe(is_unified, {GPU_B: 0})()
        assert set(readings) == {GPU_B}
        # The unified-memory check takes the CUDA ordinal, not the NVML index.
        assert seen == [0]

    def test_nvml_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            NVMLError = RuntimeError

            @staticmethod
            def nvmlInit() -> None:
                raise RuntimeError("driver wedged")

        monkeypatch.setattr("worker.gpu_availability.pynvml", Broken)
        assert NvmlDeviceProbe()() == {}


class TestMonitor:
    def _monitor(
        self, batches: list[dict[str, DeviceReading]], **cfg: Any
    ) -> GpuAvailabilityMonitor:
        config = GpuGateConfig(
            enabled=cfg.get("enabled", True),
            threshold_mib=THRESH,
            consecutive=cfg.get("consecutive", 2),
            grace_sec=0.0,
        )
        it = iter(batches)
        return GpuAvailabilityMonitor(config, lambda: next(it))

    def test_devices_are_tracked_independently(self) -> None:
        held = {GPU_A: _reading(40_000), GPU_B: _reading(5)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].available is False
        assert snapshot[GPU_B].available is True

    def test_suppressed_observation_latches(self) -> None:
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        # Nothing is probed while suppressed, so the batch list is not consumed.
        monitor.observe(False)
        assert monitor.snapshot()[GPU_A].available is False
        assert monitor.live_snapshot() == {}

    def test_total_probe_failure_clears_rather_than_latching(self) -> None:
        # Only a fresh clear reading releases a latch, and a broken probe never
        # produces one -- latching here would exclude the worker forever.
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held, {}])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        monitor.observe(True)
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

    def test_partial_failure_leaves_unread_devices_latched(self) -> None:
        both = {GPU_A: _reading(40_000), GPU_B: _reading(40_000)}
        monitor = self._monitor([both, both, {GPU_B: _reading(5)}])
        monitor.observe(True)
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].available is False, "unread device keeps its state"
        assert snapshot[GPU_B].available is False, "one clear reading is not enough"

    def test_an_unread_device_is_latched_but_not_live(self) -> None:
        # A device the probe could not read this tick may still advise the
        # dispatcher, but must not hard-refuse work nobody has confirmed since.
        both = {GPU_A: _reading(40_000), GPU_B: _reading(40_000)}
        monitor = self._monitor([both, both, {GPU_B: _reading(40_000)}])
        monitor.observe(True)
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        assert GPU_A not in monitor.live_snapshot()
        assert GPU_B in monitor.live_snapshot()

    def test_a_disabled_monitor_never_probes(self) -> None:
        # The kill switch is also enforced at construction; this is the guard that
        # survives a second construction site appearing.
        def explode() -> dict[str, DeviceReading]:
            pytest.fail("probed while disabled")

        monitor = GpuAvailabilityMonitor(GpuGateConfig(enabled=False), explode)
        monitor.observe(True)
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

    def test_free_bytes_ride_along(self) -> None:
        monitor = self._monitor([{GPU_A: _reading(5, free_bytes=1234)}])
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].free_bytes == 1234


class FakeClient:
    def __init__(self) -> None:
        self.statuses: list[tuple[WorkerStatus, dict[str, Any] | None]] = []

    def set_status(self, status: WorkerStatus, extra: dict | None = None) -> None:
        self.statuses.append((status, extra))


def _lifecycle(
    tmp_path: Path, batches: list[dict[str, DeviceReading]], grace: float = 0.0
) -> tuple[Lifecycle, GpuAvailabilityMonitor, FakeClient]:
    client = FakeClient()
    it = iter(batches)
    monitor = GpuAvailabilityMonitor(
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
    def test_availability_is_reported_without_touching_status(
        self, tmp_path: Path
    ) -> None:
        # The whole point of this change: the worker stays IDLE and keeps taking
        # CPU work while its GPU is reported as held.
        lc, _, client = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert client.statuses == []
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False

    def test_a_warm_gpu_executor_suppresses_the_reading(self, tmp_path: Path) -> None:
        # Regression guard: reading our own resident model and calling it foreign
        # is the bug 49058d7 fixed, and it must not come back by another route.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc.set_gpu_executor_probe(lambda: True)
        lc._observe_gpu()
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

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

    def test_a_finished_task_no_longer_wipes_availability(self, tmp_path: Path) -> None:
        # set_idle used to clear the gate, which is why admission once had to run
        # before set_busy. The latch now survives a task boundary.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert monitor.snapshot()[GPU_A].available is False
        lc.set_busy("tsk-1")
        lc.set_idle("tsk-1")
        assert monitor.snapshot()[GPU_A].available is False

    def test_live_availability_hides_a_stale_latch(self, tmp_path: Path) -> None:
        # Refusing a task on a latch we cannot currently confirm would fail it
        # terminally; the advisory snapshot keeps it, the live view does not.
        lc, _, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert lc.live_gpu_availability()[GPU_A].available is False
        lc.set_gpu_executor_probe(lambda: True)
        lc._observe_gpu()
        assert lc.live_gpu_availability() == {}
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False


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


class TestAdmission:
    """``_refuse_if_gpu_is_held``: the worker's own veto on a held card."""

    def _runner(
        self, availability: dict[str, DeviceAvailability], devices: int = 1
    ) -> Runner:
        lifecycle = MagicMock()
        lifecycle.live_gpu_availability.return_value = availability
        hardware = WorkerHardware(
            cpu=CPUInfo(logical_cores=8, model="CPU"),
            memory=MemoryInfo(total_bytes=64 * 1024**3),
            gpu=GpuPlatformInfo(
                driver_version=None,
                cuda_version=None,
                devices=[
                    GpuInfo(
                        index=i,
                        name="NVIDIA RTX 6000 Ada Generation",
                        uuid=f"GPU-{i}",
                        memory_total_bytes=48 * 1024**3,
                    )
                    for i in range(devices)
                ],
            ),
            network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
        )
        return Runner(
            lifecycle=lifecycle,
            task_stream=[],
            results_dir=Path("/tmp/unused"),
            hardware=hardware,
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

    def _gpu_spec(self) -> Any:
        return InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            data={"type": "list", "items": ["hi"]},
            model=ModelConfig(source=ModelSource(identifier="org/m")),
        )

    def _held(self, *uuids: str) -> dict[str, DeviceAvailability]:
        return {u: DeviceAvailability(available=False, free_bytes=0) for u in uuids}

    def test_refuses_a_gpu_task_when_the_only_device_is_held(self) -> None:
        runner = self._runner(self._held("GPU-0"))
        with pytest.raises(ExecutionError) as excinfo:
            runner._refuse_if_gpu_is_held(self._gpu_spec())
        assert excinfo.value.retryable is True, "must reroute, not fail the task"

    def test_admits_when_a_free_device_remains(self) -> None:
        runner = self._runner(self._held("GPU-0"), devices=4)
        runner._refuse_if_gpu_is_held(self._gpu_spec())

    def test_admits_a_cpu_task_onto_a_fully_held_worker(self) -> None:
        runner = self._runner(self._held("GPU-0"))
        runner._refuse_if_gpu_is_held(EchoSpecStrict(taskType=TaskType.ECHO))

    def test_a_stale_latch_does_not_refuse(self) -> None:
        # live_gpu_availability returns {} when the last reading was suppressed.
        runner = self._runner({})
        runner._refuse_if_gpu_is_held(self._gpu_spec())

    def test_nothing_held_admits(self) -> None:
        runner = self._runner(
            {"GPU-0": DeviceAvailability(available=True, free_bytes=1)}
        )
        runner._refuse_if_gpu_is_held(self._gpu_spec())


class TestClearingReachesTheServer:
    """A probe failure has to clear the server's copy, not just the worker's.

    The server latches what it was last told, so if the worker simply stopped
    mentioning a device the stale reading would sit there with nothing able to
    release it -- the worker would be held out of GPU placement indefinitely.
    """

    def test_an_empty_reading_is_still_reported(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}, {}])
        lc._observe_gpu()
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False
        lc._observe_gpu()  # probe fails
        assert lc._metrics()["gpu_availability"] == {}

    def test_a_worker_without_a_monitor_says_nothing(self, tmp_path: Path) -> None:
        # Absent key means "no opinion offered"; an empty map means "cleared".
        lc = Lifecycle(MagicMock(), 30, 120, tmp_path / "hb", cost_per_hour=0.0)
        assert "gpu_availability" not in lc._metrics()
