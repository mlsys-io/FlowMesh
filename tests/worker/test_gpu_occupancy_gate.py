"""Foreign-GPU gate: a worker whose GPU is held outside FlowMesh reports UNAVAILABLE."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared.schemas.worker import WorkerStatus
from worker.gpu_occupancy import (
    MIB,
    GateState,
    GpuGate,
    GpuGateConfig,
    NvmlUsageProbe,
    decide,
)
from worker.lifecycle import Lifecycle

THRESH = 1024


# --------------------------------------------------------------------------- #
# decide(): the pure state machine
# --------------------------------------------------------------------------- #


def test_needs_consecutive_observations_to_enter() -> None:
    s = decide(40_000, False, THRESH, 2, GateState())
    assert s == GateState(unavailable=False, used_mib=40_000, streak=1)
    s = decide(40_000, False, THRESH, 2, s)
    assert s == GateState(unavailable=True, used_mib=40_000, streak=0)


def test_needs_consecutive_observations_to_leave() -> None:
    s = decide(0, False, THRESH, 2, GateState(unavailable=True))
    assert s.unavailable is True
    s = decide(0, False, THRESH, 2, s)
    assert s == GateState(unavailable=False, used_mib=0, streak=0)


def test_single_spike_does_not_flip() -> None:
    s = decide(40_000, False, THRESH, 2, GateState())
    s = decide(10, False, THRESH, 2, s)  # back under before the streak completes
    assert s == GateState(unavailable=False, used_mib=10, streak=0)


def test_active_task_never_flips_and_resets_streak() -> None:
    # Memory used during this worker's own task is its own, never foreign.
    s = decide(40_000, True, THRESH, 2, GateState(streak=1))
    assert s == GateState(unavailable=False, used_mib=40_000, streak=0)
    s = decide(0, True, THRESH, 2, GateState(unavailable=True, streak=1))
    assert s == GateState(unavailable=True, used_mib=0, streak=0)


def test_unreadable_nvml_keeps_state() -> None:
    # Fail open: an NVML error must not take workers out of the pool,
    # nor bring an occupied one back.
    assert decide(None, False, THRESH, 2, GateState()) == GateState()
    assert decide(None, False, THRESH, 2, GateState(unavailable=True)) == GateState(
        unavailable=True
    )


def test_at_threshold_is_not_occupied() -> None:
    s = decide(THRESH, False, THRESH, 1, GateState())
    assert s.unavailable is False


def test_consecutive_one_flips_immediately() -> None:
    assert decide(40_000, False, THRESH, 1, GateState()).unavailable is True


def test_nvml_probe_skips_unified_devices_and_keeps_zero_readings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNvml:
        @staticmethod
        def nvmlInit() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetCount() -> int:
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> int:
            return index

        @staticmethod
        def nvmlDeviceGetName(handle: int) -> str:
            return "unified" if handle == 0 else "dedicated"

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle: int) -> Any:
            return SimpleNamespace(used=40_000 * MIB if handle == 0 else 0)

    monkeypatch.setattr("worker.gpu_occupancy.pynvml", FakeNvml)

    assert NvmlUsageProbe(lambda index, _name: index == 0)() == 0.0


# --------------------------------------------------------------------------- #
# Lifecycle integration: what the supervisor is actually told
# --------------------------------------------------------------------------- #


class FakeClient:
    def __init__(self) -> None:
        self.statuses: list[tuple[WorkerStatus, dict[str, Any] | None]] = []

    def set_status(self, status: WorkerStatus, extra: dict | None = None) -> None:
        self.statuses.append((status, extra))


def _lifecycle(
    tmp_path: Path, readings: list[float | None], grace: float = 0.0
) -> tuple[Lifecycle, FakeClient]:
    client = FakeClient()
    it = iter(readings)
    gpu_gate = GpuGate(
        GpuGateConfig(
            enabled=True, threshold_mib=THRESH, consecutive=2, grace_sec=grace
        ),
        lambda: next(it),
    )
    lc = Lifecycle(
        client,  # type: ignore[arg-type]
        30,
        120,
        tmp_path / "hb",
        cost_per_hour=0.0,
        gpu_gate=gpu_gate,
    )
    lc._reported = WorkerStatus.IDLE  # as after start()
    return lc, client


def test_idle_worker_goes_unavailable_then_recovers(tmp_path: Path) -> None:
    lc, client = _lifecycle(tmp_path, [44_000, 44_000, 5, 5])
    lc._evaluate_gpu_gate()
    assert client.statuses == []  # one reading is not enough
    lc._evaluate_gpu_gate()
    assert client.statuses[-1][0] is WorkerStatus.UNAVAILABLE
    assert client.statuses[-1][1] == {
        "reason": "foreign_gpu_occupancy",
        "gpu_used_mib": 44_000,
    }
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert client.statuses[-1][0] is WorkerStatus.IDLE
    assert len(client.statuses) == 2


def test_busy_worker_is_never_gated(tmp_path: Path) -> None:
    lc, client = _lifecycle(tmp_path, [])  # probe must not even be read
    lc.set_busy("tsk-1")
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert [s for s, _ in client.statuses] == [WorkerStatus.BUSY]


def test_grace_after_task_skips_checks(tmp_path: Path) -> None:
    lc, client = _lifecycle(tmp_path, [], grace=3600)
    lc.set_busy("tsk-1")
    lc.set_idle("tsk-1")  # own executor may still be releasing memory
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert [s for s, _ in client.statuses] == [WorkerStatus.BUSY, WorkerStatus.IDLE]


def test_task_end_resets_gate_to_idle(tmp_path: Path) -> None:
    lc, client = _lifecycle(tmp_path, [44_000, 44_000])
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert lc._reported is WorkerStatus.UNAVAILABLE
    # A task raced in anyway; once it ends the worker reports IDLE and the
    # gate starts over rather than silently staying UNAVAILABLE.
    lc.set_busy("tsk-2")
    lc.set_idle("tsk-2")
    assert lc._reported is WorkerStatus.IDLE
    assert lc._gpu_gate is not None
    assert lc._gpu_gate.state.unavailable is False
    assert lc._last_task_end <= time.time()


def test_resident_executor_suppresses_gate(tmp_path: Path) -> None:
    # A warm executor's own GPU memory must not read as foreign, even past the
    # grace period: idle cleanup may be disabled, keeping the executor resident
    # for the worker's whole idle stretch.
    lc, client = _lifecycle(tmp_path, [44_000, 44_000])
    resident = {"loaded": True}
    lc.set_active_executor_probe(lambda: resident["loaded"])
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert client.statuses == []  # own executor holds the card; never gated
    resident["loaded"] = False  # executor idle-cleaned up; memory now foreign
    lc._evaluate_gpu_gate()
    lc._evaluate_gpu_gate()
    assert client.statuses[-1][0] is WorkerStatus.UNAVAILABLE


def test_disabled_gate_is_inert(tmp_path: Path) -> None:
    client = FakeClient()
    gpu_gate = GpuGate(GpuGateConfig(enabled=False), lambda: pytest.fail("probed"))
    lc = Lifecycle(
        client,  # type: ignore[arg-type]
        30,
        120,
        tmp_path / "hb",
        cost_per_hour=0.0,
        gpu_gate=gpu_gate,
    )
    lc._reported = WorkerStatus.IDLE
    lc._evaluate_gpu_gate()
    assert client.statuses == []
