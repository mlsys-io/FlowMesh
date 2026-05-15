"""Tests for worker hardware satisfaction and sorting."""

from server.registries.worker import Worker, hw_satisfies
from shared.schemas.worker import SSHLimits, WorkerStatus
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)


def _worker(
    id: str = "w-1",
    gpu_count: int = 0,
    gpu_mem: int = 0,
    gpu_name: str = "A100",
    sys_mem: int = 0,
    cpu_cores: int = 4,
    gpu_memory_is_unified: bool = False,
    gpu_shared_memory_total_bytes: int | None = None,
    ssh_limits: SSHLimits | None = None,
) -> Worker:
    devices = [
        GpuInfo(index=i, name=gpu_name, uuid=f"GPU-{i}", memory_total_bytes=gpu_mem)
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
        ssh_limits=ssh_limits,
    )


def _task(
    gpu_count: int | None = None,
    gpu_memory: str | None = None,
    gpu_type: str | None = None,
    cpu: int | None = None,
    memory: str | None = None,
) -> TaskEnvelopeStrict:
    gpu_req = None
    if gpu_count is not None or gpu_memory or gpu_type:
        gpu_req = GPURequirements(count=gpu_count, memory=gpu_memory, type=gpu_type)
    hw_req = None
    if gpu_req or cpu is not None or memory:
        hw_req = HardwareRequirements(gpu=gpu_req, cpu=cpu, memory=memory)
    resources = ResourcesSpec(hardware=hw_req) if hw_req else None
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "spec": {
                "taskType": "echo",
                "resources": resources.model_dump() if resources else None,
            },
        }
    )


class TestHwSatisfies:
    def test_no_requirements(self) -> None:
        assert hw_satisfies(_worker(), _task()) is True

    def test_gpu_count_satisfied(self) -> None:
        w = _worker(gpu_count=4, gpu_mem=80_000_000_000)
        t = _task(gpu_count=2)
        assert hw_satisfies(w, t) is True

    def test_gpu_count_not_satisfied(self) -> None:
        w = _worker(gpu_count=1)
        t = _task(gpu_count=4)
        assert hw_satisfies(w, t) is False

    def test_gpu_memory_satisfied(self) -> None:
        w = _worker(gpu_count=1, gpu_mem=80_000_000_000)
        t = _task(gpu_count=1, gpu_memory="40GB")
        assert hw_satisfies(w, t) is True

    def test_gpu_memory_not_satisfied(self) -> None:
        w = _worker(gpu_count=1, gpu_mem=16_000_000_000)
        t = _task(gpu_count=1, gpu_memory="40GB")
        assert hw_satisfies(w, t) is False

    def test_gpu_memory_satisfied_by_unified_pool(self) -> None:
        w = _worker(
            gpu_count=1,
            gpu_mem=0,
            gpu_name="NVIDIA GB10",
            sys_mem=128 * (1 << 30),
            gpu_memory_is_unified=True,
            gpu_shared_memory_total_bytes=128 * (1 << 30),
        )
        t = _task(gpu_count=1, gpu_memory="40GB")
        assert hw_satisfies(w, t) is True

    def test_gpu_memory_not_satisfied_by_unified_pool(self) -> None:
        w = _worker(
            gpu_count=1,
            gpu_mem=0,
            gpu_name="NVIDIA GB10",
            sys_mem=32 * (1 << 30),
            gpu_memory_is_unified=True,
            gpu_shared_memory_total_bytes=32 * (1 << 30),
        )
        t = _task(gpu_count=1, gpu_memory="40GB")
        assert hw_satisfies(w, t) is False

    def test_gpu_type_match(self) -> None:
        w = _worker(gpu_count=1, gpu_name="NVIDIA A100-SXM4-80GB")
        t = _task(gpu_count=1, gpu_type="A100")
        assert hw_satisfies(w, t) is True

    def test_gpu_type_mismatch(self) -> None:
        w = _worker(gpu_count=1, gpu_name="NVIDIA T4")
        t = _task(gpu_count=1, gpu_type="A100")
        assert hw_satisfies(w, t) is False

    def test_null_hardware_fails_gpu(self) -> None:
        w = Worker(
            id="w-1",
            namespace="ns",
            cluster="cl",
            node_id="g-1",
            node_alias="g",
            status=WorkerStatus.IDLE,
            hardware=None,
        )
        t = _task(gpu_count=1)
        assert hw_satisfies(w, t) is False

    def test_cpu_requirement(self) -> None:
        w = _worker(cpu_cores=8)
        t = _task(cpu=4)
        assert hw_satisfies(w, t) is True

    def test_cpu_not_satisfied(self) -> None:
        w = _worker(cpu_cores=2)
        t = _task(cpu=8)
        assert hw_satisfies(w, t) is False


def _ssh_task(cpu: int | None = None, memory: str | None = None) -> TaskEnvelopeStrict:
    hw_req = None
    if cpu is not None or memory is not None:
        hw_req = HardwareRequirements(cpu=cpu, memory=memory)
    resources = ResourcesSpec(hardware=hw_req) if hw_req else None
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "spec": {
                "taskType": "ssh",
                "interactive": False,
                "image": "x",
                "command": ["true"],
                "resources": resources.model_dump() if resources else None,
            },
        }
    )


class TestHwSatisfiesSSHLimits:
    def test_ssh_cap_below_request_filters_worker(self) -> None:
        w = _worker(
            cpu_cores=32,
            sys_mem=64 * 1024**3,
            ssh_limits=SSHLimits(max_cpu_cores=2.0),
        )
        t = _ssh_task(cpu=8)
        assert hw_satisfies(w, t) is False

    def test_ssh_cap_above_request_passes(self) -> None:
        w = _worker(
            cpu_cores=32,
            sys_mem=64 * 1024**3,
            ssh_limits=SSHLimits(max_cpu_cores=16.0),
        )
        t = _ssh_task(cpu=8)
        assert hw_satisfies(w, t) is True

    def test_ssh_memory_cap_filters(self) -> None:
        w = _worker(
            cpu_cores=32,
            sys_mem=64 * 1024**3,
            ssh_limits=SSHLimits(max_memory_bytes=2 * 1024**3),
        )
        t = _ssh_task(memory="4Gi")
        assert hw_satisfies(w, t) is False

    def test_ssh_cap_ignored_for_non_ssh_tasks(self) -> None:
        # Even if ssh_limits would filter out the worker for SSH, non-SSH
        # tasks should see the full physical hardware.
        w = _worker(
            cpu_cores=32,
            ssh_limits=SSHLimits(max_cpu_cores=2.0),
        )
        t = _task(cpu=8)
        assert hw_satisfies(w, t) is True

    def test_no_ssh_cap_behaves_as_before(self) -> None:
        w = _worker(cpu_cores=32, sys_mem=64 * 1024**3)
        t = _ssh_task(cpu=8, memory="4Gi")
        assert hw_satisfies(w, t) is True
