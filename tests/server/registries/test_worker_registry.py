"""Tests for worker hardware satisfaction and sorting."""

import json

from server.registries.worker import (
    Worker,
    _parse_worker_from_redis,
    capability_satisfies,
    gpu_available_for,
    hw_satisfies,
)
from server.schemas.node import NodeWorkerStatus
from shared.schemas.worker import SSHLimits, WorkerCapabilities, WorkerStatus
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.task_type import TaskType
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
    capabilities: WorkerCapabilities | None = None,
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
        capabilities=capabilities or WorkerCapabilities(),
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


class TestCapabilitySatisfies:
    def test_task_requires_advertised_task_type(self) -> None:
        w = _worker(
            capabilities=WorkerCapabilities(
                supported_task_types=frozenset({TaskType.ECHO})
            )
        )
        assert capability_satisfies(w, _ssh_task()) is False

    def test_task_accepts_worker_advertising_task_type(self) -> None:
        w = _worker(
            capabilities=WorkerCapabilities(
                supported_task_types=frozenset({TaskType.SSH})
            )
        )
        assert capability_satisfies(w, _ssh_task()) is True

    def test_default_worker_supports_nothing(self) -> None:
        # An unreporting worker parses with the strict (empty) default.
        assert capability_satisfies(_worker(), _ssh_task()) is False
        assert capability_satisfies(_worker(), _task(cpu=2)) is False

    def test_gate_covers_all_task_types(self) -> None:
        w = _worker(
            capabilities=WorkerCapabilities(
                supported_task_types=frozenset({TaskType.ECHO})
            )
        )
        assert capability_satisfies(w, _task(cpu=2)) is True
        assert capability_satisfies(w, _ssh_task()) is False


class TestParseCapabilities:
    def test_capabilities_round_trip(self) -> None:
        raw = {
            "status": "IDLE",
            "capabilities_json": WorkerCapabilities(
                supported_task_types=frozenset({TaskType.SSH, TaskType.ECHO})
            ).model_dump_json(),
        }
        w = _parse_worker_from_redis("w-1", raw)
        assert w is not None
        assert w.capabilities.supported_task_types == {TaskType.SSH, TaskType.ECHO}

    def test_missing_capabilities_defaults_strict(self) -> None:
        w = _parse_worker_from_redis("w-1", {"status": "IDLE"})
        assert w is not None
        assert w.capabilities.supported_task_types == frozenset()


class TestParseVersion:
    def test_version_round_trip(self) -> None:
        w = _parse_worker_from_redis("w-1", {"status": "IDLE", "version": "0.1.0"})
        assert w is not None
        assert w.version == "0.1.0"

    def test_missing_version_defaults_none(self) -> None:
        w = _parse_worker_from_redis("w-1", {"status": "IDLE"})
        assert w is not None
        assert w.version is None


class TestParseStatus:
    def test_unavailable_round_trips(self) -> None:
        w = _parse_worker_from_redis("w-1", {"status": "UNAVAILABLE"})
        assert w is not None
        assert w.status is WorkerStatus.UNAVAILABLE

    def test_unknown_status_degrades_instead_of_raising(self) -> None:
        # A worker on a newer build may report a status this host does not
        # know. Raising would break every registry read of that worker,
        # including the dispatcher's candidate scan.
        w = _parse_worker_from_redis("w-1", {"status": "SOME_FUTURE_STATE"})
        assert w is not None
        assert w.status is WorkerStatus.UNKNOWN

    def test_node_worker_status_degrades_instead_of_raising(self) -> None:
        # Node worker listings validate an unregistered worker's status
        # straight from the node's report, so an unrecognised value must
        # degrade rather than 500 the endpoint.
        assert NodeWorkerStatus("RUNNING") is NodeWorkerStatus.UNKNOWN


class TestParseGpuOccupancy:
    def _raw(self, occupancy: dict | None = None) -> dict:
        hardware = WorkerHardware(
            cpu=CPUInfo(logical_cores=16, model="AMD EPYC 7543"),
            memory=MemoryInfo(total_bytes=64 * 1024**3),
            gpu=GpuPlatformInfo(
                driver_version="550.0",
                cuda_version="12.4",
                devices=[
                    GpuInfo(
                        index=0,
                        name="NVIDIA RTX 6000 Ada Generation",
                        uuid="GPU-held",
                        memory_total_bytes=48 * 1024**3,
                    ),
                    GpuInfo(
                        index=1,
                        name="NVIDIA RTX 6000 Ada Generation",
                        uuid="GPU-free",
                        memory_total_bytes=48 * 1024**3,
                    ),
                ],
            ),
            network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
        )
        raw = {"status": "IDLE", "hardware_json": hardware.model_dump_json()}
        if occupancy is not None:
            raw["gpu_occupancy_json"] = json.dumps(occupancy)
        return raw

    def test_occupancy_joins_onto_devices_by_uuid(self) -> None:
        raw = self._raw(
            {
                "GPU-held": {"unavailable": True, "free_bytes": 19 * 1024**2},
                "GPU-free": {"unavailable": False, "free_bytes": 47 * 1024**3},
            }
        )
        w = _parse_worker_from_redis("w-1", raw)
        assert w is not None and w.hardware is not None
        held, free = w.hardware.gpu.devices
        assert held.gpu_unavailable is True
        assert held.memory_free_bytes == 19 * 1024**2
        assert free.gpu_unavailable is False

    def test_a_device_the_worker_did_not_mention_stays_unknown(self) -> None:
        w = _parse_worker_from_redis(
            "w-1", self._raw({"GPU-held": {"unavailable": True}})
        )
        assert w is not None and w.hardware is not None
        held, free = w.hardware.gpu.devices
        assert held.gpu_unavailable is True
        assert free.gpu_unavailable is None, "unreported means unknown, not free"

    def test_no_occupancy_field_leaves_every_device_unknown(self) -> None:
        # An older worker, or one that has never taken a usable reading.
        w = _parse_worker_from_redis("w-1", self._raw())
        assert w is not None and w.hardware is not None
        assert all(d.gpu_unavailable is None for d in w.hardware.gpu.devices)

    def test_malformed_occupancy_is_ignored(self) -> None:
        w = _parse_worker_from_redis("w-1", self._raw({"GPU-held": "nonsense"}))
        assert w is not None and w.hardware is not None
        assert w.hardware.gpu.devices[0].gpu_unavailable is None


def _held(worker: Worker, *indices: int) -> Worker:
    """Mark the given device indices as held by another tenant."""
    assert worker.hardware is not None
    for index in indices:
        worker.hardware.gpu.devices[index].gpu_unavailable = True
    return worker


def _inference_task() -> TaskEnvelopeStrict:
    """A GPU task that declares no resources -- the shape that caused the incident."""
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "spec": {
                "taskType": "inference",
                "data": {"type": "list", "items": ["hi"]},
                "model": {"source": {"identifier": "org/m"}},
            },
        }
    )


class TestGpuAvailableFor:
    def test_held_device_is_not_offered(self) -> None:
        worker = _held(_worker(gpu_count=1, gpu_mem=48 * 1024**3), 0)
        assert gpu_available_for(worker, _task(gpu_count=1)) is False

    def test_a_free_sibling_still_is(self) -> None:
        # The whole point of per-device: one held card must not write off the box.
        worker = _held(_worker(gpu_count=4, gpu_mem=48 * 1024**3), 0)
        assert gpu_available_for(worker, _task(gpu_count=1)) is True

    def test_not_enough_free_devices(self) -> None:
        worker = _held(_worker(gpu_count=2, gpu_mem=48 * 1024**3), 0)
        assert gpu_available_for(worker, _task(gpu_count=2)) is False

    def test_undeclared_gpu_task_is_still_filtered(self) -> None:
        # hw_satisfies never reaches a GPU check for this task, which is why the
        # filter cannot live there.
        worker = _held(_worker(gpu_count=1, gpu_mem=48 * 1024**3), 0)
        task = _inference_task()
        assert hw_satisfies(worker, task) is True
        assert gpu_available_for(worker, task) is False

    def test_cpu_task_is_unaffected(self) -> None:
        # Goal 1: the worker keeps earning its keep on CPU work.
        worker = _held(_worker(gpu_count=1, gpu_mem=48 * 1024**3), 0)
        assert gpu_available_for(worker, _task(cpu=2)) is True

    def test_hw_satisfies_is_not_changed_by_occupancy(self) -> None:
        # satisfying_workers must keep the worker, or the task fails as
        # unschedulable instead of waiting for the card to free up.
        worker = _held(_worker(gpu_count=1, gpu_mem=48 * 1024**3), 0)
        assert hw_satisfies(worker, _task(gpu_count=1)) is True


class TestGpuAvailableForOnlySubtracts:
    def test_cpu_only_worker_is_untouched(self) -> None:
        # No devices reported at all: this predicate must never be stricter than
        # hw_satisfies on a worker it knows nothing about.
        assert gpu_available_for(_worker(gpu_count=0), _inference_task()) is True

    def test_worker_reporting_no_occupancy_is_untouched(self) -> None:
        worker = _worker(gpu_count=1, gpu_mem=48 * 1024**3)
        assert worker.hardware is not None
        assert all(d.gpu_unavailable is None for d in worker.hardware.gpu.devices)
        assert gpu_available_for(worker, _inference_task()) is True

    def test_unified_memory_worker_is_untouched(self) -> None:
        # The probe skips unified devices, so they never report occupied.
        worker = _worker(
            gpu_count=1,
            gpu_mem=0,
            gpu_memory_is_unified=True,
            gpu_shared_memory_total_bytes=128 * 1024**3,
        )
        assert gpu_available_for(worker, _task(gpu_memory="40Gi")) is True

    def test_unified_pool_still_reachable_when_another_device_is_held(self) -> None:
        worker = _held(
            _worker(
                gpu_count=2,
                gpu_mem=0,
                gpu_memory_is_unified=True,
                gpu_shared_memory_total_bytes=128 * 1024**3,
            ),
            0,
        )
        assert gpu_available_for(worker, _task(gpu_memory="40Gi")) is True
