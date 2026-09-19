"""Helpers for parsing GPU requirement specs and matching them against devices."""

import re

from shared.tasks.components.resources import GPURequirements
from shared.tasks.worker_message import GpuInfo, WorkerHardware
from shared.utils.parsing import parse_mem_to_bytes

_GPU_TYPE_WILDCARDS = frozenset({"", "any", "auto", "*"})


def normalize_gpu_type(value: str | None) -> str | None:
    """Lowercase the type; ``None`` for wildcards (``''``/``any``/``auto``/``*``)."""
    if value is None:
        return None
    normalized = value.strip().lower()
    return None if normalized in _GPU_TYPE_WILDCARDS else normalized


def gpu_type_pattern(value: str | None) -> re.Pattern[str] | None:
    """Case-insensitive substring matcher, or ``None`` for wildcard."""
    normalized = normalize_gpu_type(value)
    if normalized is None:
        return None
    return re.compile(re.escape(normalized), re.IGNORECASE)


def parse_gpu_memory_bytes(value: str | int | float | None) -> int | None:
    """Parse ``GPURequirements.memory`` (str / int / float / None) to bytes.

    Returns ``None`` for ``None`` and for unparsable strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_mem_to_bytes(value)
    return int(value)


# A card with less than this free is treated as claimed by something outside
# FlowMesh and is not offered to the scheduler. Nothing we run fits in under a
# gigabyte -- vLLM cannot even build its KV cache -- so excluding such a device
# can never cost us a placement that would have succeeded.
GPU_CLAIMED_FREE_FLOOR_BYTES = 1 << 30


def gpu_device_is_claimed(device: GpuInfo) -> bool:
    """True when another tenant already holds this GPU.

    FlowMesh does not allocate GPUs on these nodes; the NVIDIA device plugin
    hands whole cards to Kubernetes pods and tells FlowMesh nothing. A worker
    therefore reports itself IDLE with a card that is entirely spoken for, and
    the scheduler -- which counted only its own tasks -- kept placing work on it.
    Free VRAM is the one signal visible from inside the worker's container: it
    cannot see pod identity, but it can see that the memory is gone.

    Unknown (``None``) means "not reported", NOT "claimed". Older workers and
    hosts without NVML must keep scheduling exactly as they did.
    """
    free = device.memory_free_bytes
    if free is None:
        return False
    return free < GPU_CLAIMED_FREE_FLOOR_BYTES


def gpu_device_matches(
    device: GpuInfo,
    *,
    type_pattern: re.Pattern[str] | None = None,
    min_memory_bytes: int | None = None,
) -> bool:
    """Per-device predicate; ``None`` arg means 'no constraint'."""
    if type_pattern is not None and not type_pattern.search(device.name or ""):
        return False
    # Checked BEFORE the memory constraint, and independently of it. The jobs
    # that exposed this carried no `gpu_memory` requirement at all, so a purely
    # threshold-based test short-circuits on `min_memory_bytes is None` and
    # admits a card with 19 MiB free.
    if gpu_device_is_claimed(device):
        return False
    if min_memory_bytes is None:
        return True
    # Prefer what is actually available; fall back to total when the worker
    # does not report free memory.
    available = device.memory_free_bytes
    if available is None:
        available = device.memory_total_bytes or 0
    return available >= min_memory_bytes


def unified_gpu_memory_satisfies(
    hw: WorkerHardware, required_memory_bytes: int, required_count: int
) -> bool:
    """Pessimistic per-slot share of a unified GPU/system memory pool.

    Returns ``True`` only when ``hw.gpu.memory_is_unified`` and
    ``shared_memory_total_bytes / required_count >= required_memory_bytes``.
    """
    if not hw.gpu.memory_is_unified:
        return False
    shared_total = hw.gpu.shared_memory_total_bytes or 0
    if shared_total <= 0:
        return False
    per_gpu_share = shared_total / max(required_count, 1)
    return per_gpu_share >= required_memory_bytes


def select_matching_gpu_indices(
    devices: list[GpuInfo],
    gpu_req: GPURequirements,
    *,
    limit: int | None = None,
) -> list[int]:
    """Indices of devices that individually pass ``gpu_req``'s type + memory.

    Stops after ``limit`` matches when set.
    """
    if limit is not None and limit <= 0:
        return []
    type_pattern = gpu_type_pattern(gpu_req.type)
    min_memory_bytes = (
        parse_gpu_memory_bytes(gpu_req.memory) if gpu_req.memory else None
    )
    result: list[int] = []
    for idx, device in enumerate(devices):
        if not gpu_device_matches(
            device,
            type_pattern=type_pattern,
            min_memory_bytes=min_memory_bytes,
        ):
            continue
        result.append(idx)
        if limit is not None and len(result) >= limit:
            break
    return result
