"""Tests for worker selection strategies."""

from server.dispatcher.worker_selector import (
    _collect_worker_metrics,
    _stable_jitter,
    select_worker,
)
from server.registries.worker import Worker
from shared.schemas.worker import WorkerStatus
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)


def _worker(
    id: str,
    gpu_count: int = 0,
    gpu_mem: int = 0,
    sys_mem: int = 0,
    cpu_cores: int = 1,
    cost: float = 1.0,
    gpu_memory_is_unified: bool = False,
    gpu_shared_memory_total_bytes: int | None = None,
) -> Worker:
    devices = [
        GpuInfo(index=i, name="GPU", uuid=f"GPU-{i}", memory_total_bytes=gpu_mem)
        for i in range(gpu_count)
    ]
    hw = WorkerHardware(
        cpu=CPUInfo(logical_cores=cpu_cores, model="CPU"),
        memory=MemoryInfo(total_bytes=sys_mem),
        gpu=GpuPlatformInfo(
            driver_version=None,
            cuda_version=None,
            devices=devices,
            memory_is_unified=gpu_memory_is_unified,
            shared_memory_total_bytes=gpu_shared_memory_total_bytes,
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )
    return Worker(
        id=id,
        namespace="ns",
        cluster="cl",
        node_id="g-1",
        node_alias="g",
        status=WorkerStatus.IDLE,
        hardware=hw,
        cost_per_hour=cost,
    )


class TestSelectWorker:
    def test_empty_pool(self) -> None:
        worker, _ = select_worker([], strategy="best_fit")
        assert worker is None

    def test_first_fit(self) -> None:
        pool = [_worker("w-1"), _worker("w-2")]
        worker, _ = select_worker(pool, strategy="first_fit")
        assert worker is not None
        assert worker.id == "w-1"

    def test_best_fit_prefers_higher_throughput(self) -> None:
        w_small = _worker("w-small", gpu_count=1, gpu_mem=16_000_000_000)
        w_big = _worker("w-big", gpu_count=4, gpu_mem=80_000_000_000)
        worker, _ = select_worker([w_small, w_big], strategy="best_fit", task_id="t-1")
        assert worker is not None
        assert worker.id == "w-big"

    def test_best_fit_lambda_weighting(self) -> None:
        # Low lambda → cost matters more; cheap worker should win
        w_cheap = _worker("w-cheap", gpu_count=1, cost=0.5)
        w_expensive = _worker("w-expensive", gpu_count=2, cost=10.0)
        worker, _ = select_worker(
            [w_cheap, w_expensive],
            strategy="best_fit",
            task_id="t-1",
            lambda_overrides={"inference": 0.0},
            task_category="inference",
        )
        assert worker is not None
        assert worker.id == "w-cheap"

    def test_min_satisfying(self) -> None:
        w_small = _worker("w-small", gpu_count=1)
        w_big = _worker("w-big", gpu_count=4)
        worker, _ = select_worker(
            [w_small, w_big],
            strategy="min_satisfying",
            task_id="t-1",
        )
        assert worker is not None
        # min_satisfying prefers least capacity → small
        assert worker.id == "w-small"

    def test_deterministic_tiebreak(self) -> None:
        pool = [_worker(f"w-{i}") for i in range(5)]
        w1, _ = select_worker(pool, strategy="best_fit", task_id="t-1")
        w2, _ = select_worker(pool, strategy="best_fit", task_id="t-1")
        assert w1 is not None and w2 is not None
        assert w1.id == w2.id

    def test_unknown_strategy_fallback(self) -> None:
        pool = [_worker("w-1")]
        worker, _ = select_worker(pool, strategy="nonexistent", task_id="t-1")
        assert worker is not None


class TestStableJitter:
    def test_determinism(self) -> None:
        j1 = _stable_jitter("t-1", "w-1", 1e-3)
        j2 = _stable_jitter("t-1", "w-1", 1e-3)
        assert j1 == j2

    def test_different_inputs(self) -> None:
        j1 = _stable_jitter("t-1", "w-1", 1e-3)
        j2 = _stable_jitter("t-1", "w-2", 1e-3)
        assert j1 != j2

    def test_zero_magnitude(self) -> None:
        assert _stable_jitter("t-1", "w-1", 0.0) == 0.0


class TestCollectMetrics:
    def test_missing_hardware(self) -> None:
        w = Worker(
            id="w-1",
            namespace="ns",
            cluster="cl",
            node_id="g-1",
            node_alias="g",
            status=WorkerStatus.IDLE,
            hardware=None,
        )
        m = _collect_worker_metrics(w)
        assert m.get("gpu_count", 0) == 0
        assert m.get("vram_total", 0) == 0
        assert m.get("sys_ram", 0) == 0
        assert m.get("cpu_cores", 0) == 0

    def test_unified_memory_worker_does_not_double_count_vram(self) -> None:
        w = _worker(
            "w-uma",
            gpu_count=1,
            gpu_mem=0,
            sys_mem=128 * (1 << 30),
            gpu_memory_is_unified=True,
            gpu_shared_memory_total_bytes=128 * (1 << 30),
        )
        m = _collect_worker_metrics(w)
        assert m["gpu_count"] == 1.0
        assert m["vram_gb"] == 0.0
        assert m["shared_gpu_mem_gb"] == 128.0
        assert m["sys_ram_gb"] == 128.0
        assert m["throughput"] == 100.0 + 128.0 + 0.5

    def test_best_fit_prefers_unified_gpu_memory_over_small_vram(self) -> None:
        w_small = _worker(
            "w-small", gpu_count=1, gpu_mem=16 * (1 << 30), sys_mem=64 * (1 << 30)
        )
        w_unified = _worker(
            "w-unified",
            gpu_count=1,
            gpu_mem=0,
            sys_mem=200 * (1 << 30),
            gpu_memory_is_unified=True,
            gpu_shared_memory_total_bytes=200 * (1 << 30),
        )
        worker, _ = select_worker(
            [w_small, w_unified], strategy="best_fit", task_id="t-1"
        )
        assert worker is not None
        assert worker.id == "w-unified"
