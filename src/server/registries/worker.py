import json
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from shared.schemas.command import (
    InterruptMessage,
    StopMessage,
    TaskMessage,
)
from shared.schemas.worker import SSHLimits, WorkerCapabilities
from shared.tasks import TaskEnvelope
from shared.tasks.components.resources import GPURequirements
from shared.tasks.specs import SSHSpecStrict, SSHSpecTemplate
from shared.tasks.worker_message import (
    WorkerHardware,
    WorkerStatus,
    WorkerTaskMessage,
)
from shared.utils import new_worker_id, now_iso, parse_mem_to_bytes
from shared.utils.hardware import (
    normalize_gpu_type,
    parse_gpu_memory_bytes,
    select_matching_gpu_indices,
    unified_gpu_memory_satisfies,
)

from ..clients.redis import (
    WORKER_EVENT_CHANNEL,
    WORKER_ID_SEQ_KEY,
    WORKERS_SET_KEY,
    RedisClient,
    node_dispatch_channel,
    worker_hb_key,
    worker_key,
)


class Worker(BaseModel):
    id: str = Field(description="Worker identifier.")
    alias: str | None = Field(
        default=None, description="Optional human-readable alias."
    )
    namespace: str = Field(description="Worker namespace.")
    cluster: str = Field(description="Worker cluster.")
    node_id: str = Field(description="Owning node identifier.")
    node_alias: str = Field(description="Owning node alias.")
    version: str | None = Field(default=None, description="Worker version.")
    status: WorkerStatus = Field(
        default=WorkerStatus.UNKNOWN, description="Worker status."
    )
    started_at: str | None = Field(default=None, description="Start timestamp.")
    pid: int | None = Field(default=None, description="Worker process ID.")
    env: dict[str, Any] = Field(default_factory=dict, description="Runtime metadata.")
    hardware: WorkerHardware | None = Field(
        default=None, description="Hardware metadata."
    )
    capabilities: WorkerCapabilities = Field(
        default_factory=WorkerCapabilities,
        description="Task capabilities the worker advertises.",
    )
    ssh_limits: SSHLimits | None = Field(
        default=None, description="Configured ceiling on SSH session resources."
    )
    tags: list[str] = Field(default_factory=list, description="Worker tags.")
    last_seen: str | None = Field(default=None, description="Last heartbeat timestamp.")
    cached_models: list[str] = Field(
        default_factory=list, description="Cached model identifiers."
    )
    cached_datasets: list[str] = Field(
        default_factory=list, description="Cached dataset identifiers."
    )
    cache_updated_ts: str | None = Field(
        default=None, description="Cache metadata update timestamp."
    )
    cost_per_hour: float | None = Field(
        default=None, description="Estimated hourly cost."
    )


class WorkerInfo(Worker):
    stale: bool = Field(description="Whether the worker heartbeat is stale.")


class WorkerRegistry:
    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    # ------------------------------------------------------------------ #
    # Worker lifecycle helpers
    # ------------------------------------------------------------------ #

    def register_worker(
        self,
        node_id: str,
        node_alias: str,
        worker_meta: dict[str, Any],
    ) -> str:
        worker_id = self._allocate_worker_id()
        worker_meta["id"] = worker_id
        worker_meta["node_id"] = node_id
        worker_meta["node_alias"] = node_alias
        with self._rds.sync.control_pipeline() as pipe:
            pipe.sadd(WORKERS_SET_KEY, worker_id)
            pipe.hset(worker_key(worker_id), mapping=worker_meta)
            pipe.execute()
        return worker_id

    async def register_worker_async(
        self,
        node_id: str,
        node_alias: str,
        worker_meta: dict[str, Any],
    ) -> str:
        worker_id = await self._allocate_worker_id_async()
        worker_meta["id"] = worker_id
        worker_meta["node_id"] = node_id
        worker_meta["node_alias"] = node_alias
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.sadd(WORKERS_SET_KEY, worker_id)
            pipe.hset(worker_key(worker_id), mapping=worker_meta)
            await pipe.execute()
        return worker_id

    def update_worker_hb(self, worker_id: str, ts: str, ttl_sec: int) -> None:
        with self._rds.sync.control_pipeline() as pipe:
            pipe.setex(worker_hb_key(worker_id), ttl_sec, ts)
            pipe.hset(worker_key(worker_id), mapping={"last_seen": ts})
            pipe.execute()

    async def update_worker_hb_async(
        self, worker_id: str, ts: str, ttl_sec: int
    ) -> None:
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.setex(worker_hb_key(worker_id), ttl_sec, ts)
            pipe.hset(worker_key(worker_id), mapping={"last_seen": ts})
            await pipe.execute()

    def set_worker_status(
        self,
        worker_id: str,
        status: WorkerStatus,
        ts: str,
        extra: dict[str, Any] | None,
    ) -> None:
        mapping = {"status": status.value, "last_seen": ts}
        if extra:
            mapping.update({f"extra_{k}": str(v) for k, v in extra.items()})
        with self._rds.sync.control_pipeline() as pipe:
            pipe.hset(worker_key(worker_id), mapping=mapping)
            pipe.execute()

    async def set_worker_status_async(
        self,
        worker_id: str,
        status: WorkerStatus,
        ts: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        mapping = {"status": status.value, "last_seen": ts}
        if extra:
            mapping.update({f"extra_{k}": str(v) for k, v in extra.items()})
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.hset(worker_key(worker_id), mapping=mapping)
            await pipe.execute()

    def unregister_workers(self, *worker_ids: str) -> None:
        with self._rds.sync.control_pipeline() as pipe:
            pipe.srem(WORKERS_SET_KEY, *worker_ids)
            pipe.delete(*(worker_key(worker_id) for worker_id in worker_ids))
            pipe.delete(*(worker_hb_key(worker_id) for worker_id in worker_ids))
            pipe.execute()

    async def unregister_workers_async(self, *worker_ids: str) -> None:
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.srem(WORKERS_SET_KEY, *worker_ids)
            pipe.delete(*(worker_key(worker_id) for worker_id in worker_ids))
            pipe.delete(*(worker_hb_key(worker_id) for worker_id in worker_ids))
            await pipe.execute()

    # ------------------------------------------------------------------ #
    # Worker query helpers
    # ------------------------------------------------------------------ #

    def get_worker_ids(self) -> set[str]:
        return self._rds.sync.set_members(WORKERS_SET_KEY)

    async def get_worker_ids_async(self) -> set[str]:
        return await self._rds.asyncio.set_members(WORKERS_SET_KEY)

    def get_worker(self, worker_id: str) -> Worker | None:
        raw = self._rds.sync.hash_getall(worker_key(worker_id))
        return _parse_worker_from_redis(worker_id, raw)

    async def get_worker_async(self, worker_id: str) -> Worker | None:
        raw = await self._rds.asyncio.hash_getall(worker_key(worker_id))
        return _parse_worker_from_redis(worker_id, raw)

    def get_workers(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        with self._rds.sync.control_pipeline() as pipe:
            for worker_id in worker_ids:
                pipe.hgetall(worker_key(worker_id))
            raws: list[dict] = pipe.execute()
        return [
            _parse_worker_from_redis(worker_id, raw)
            for worker_id, raw in zip(worker_ids, raws)
        ]

    async def get_workers_async(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        async with self._rds.asyncio.control_pipeline() as pipe:
            for worker_id in worker_ids:
                pipe.hgetall(worker_key(worker_id))
            raws: list[dict] = await pipe.execute()
        return [
            _parse_worker_from_redis(worker_id, raw)
            for worker_id, raw in zip(worker_ids, raws)
        ]

    def is_worker_stale(self, worker_id: str) -> bool:
        ttl = self._rds.sync.ttl(worker_hb_key(worker_id))
        return ttl is None or ttl < 0

    async def is_worker_stale_async(self, worker_id: str) -> bool:
        ttl = await self._rds.asyncio.ttl(worker_hb_key(worker_id))
        return ttl is None or ttl < 0

    def get_worker_heartbeat(self, worker_id: str) -> str | None:
        return self._rds.sync.get(worker_hb_key(worker_id))

    async def get_worker_heartbeat_async(self, worker_id: str) -> str | None:
        return await self._rds.asyncio.get(worker_hb_key(worker_id))

    def worker_exists(self, worker_id: str) -> bool:
        return self._rds.sync.exists(worker_key(worker_id))

    async def worker_exists_async(self, worker_id: str) -> bool:
        return await self._rds.asyncio.exists(worker_key(worker_id))

    def update_worker_status(self, worker_id: str, status: WorkerStatus) -> None:
        ts = now_iso()
        payload = {
            "type": "STATUS",
            "worker_id": worker_id,
            "status": status.value,
            "ts": ts,
        }
        self._rds.sync.hash_set(
            worker_key(worker_id), {"status": status.value, "last_seen": ts}
        )
        self._rds.sync.publish_telemetry(
            WORKER_EVENT_CHANNEL, json.dumps(payload, ensure_ascii=False)
        )

    async def update_worker_status_async(
        self, worker_id: str, status: WorkerStatus
    ) -> None:
        ts = now_iso()
        payload = {
            "type": "STATUS",
            "worker_id": worker_id,
            "status": status.value,
            "ts": ts,
        }
        await self._rds.asyncio.hash_set(
            worker_key(worker_id),
            {"status": status.value, "last_seen": ts},
        )
        await self._rds.asyncio.publish_telemetry(
            WORKER_EVENT_CHANNEL, json.dumps(payload, ensure_ascii=False)
        )

    def list_workers(self) -> list[WorkerInfo]:
        results: list[WorkerInfo] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker:
                continue
            stale = self.is_worker_stale(worker_id)
            if not worker.last_seen:
                hb_ts = self.get_worker_heartbeat(worker_id)
                if hb_ts:
                    worker.last_seen = hb_ts
            results.append(
                WorkerInfo(
                    **worker.model_dump(),
                    stale=stale,
                )
            )
        return results

    async def list_workers_async(self) -> list[WorkerInfo]:
        results: list[WorkerInfo] = []
        for worker_id in await self.get_worker_ids_async():
            worker = await self.get_worker_async(worker_id)
            if not worker:
                continue
            stale = await self.is_worker_stale_async(worker_id)
            if not worker.last_seen:
                hb_ts = await self.get_worker_heartbeat_async(worker_id)
                if hb_ts:
                    worker.last_seen = hb_ts
            results.append(
                WorkerInfo(
                    **worker.model_dump(),
                    stale=stale,
                )
            )
        return results

    def record_worker_cache(
        self,
        worker_id: str,
        models: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
    ) -> None:
        models_list = [v for v in (models or []) if _normalize_cache_value(v)]
        datasets_list = [v for v in (datasets or []) if _normalize_cache_value(v)]
        if not models_list and not datasets_list:
            return

        if not self.worker_exists(worker_id):
            return

        try:
            current_models_json, current_datasets_json = self._rds.sync.hash_mget(
                worker_key(worker_id), ["cache_models_json", "cache_datasets_json"]
            )
        except Exception:
            return

        try:
            current_models = (
                json.loads(current_models_json) if current_models_json else []
            )
            if not isinstance(current_models, list):
                current_models = []
        except Exception:
            current_models = []

        try:
            current_datasets = (
                json.loads(current_datasets_json) if current_datasets_json else []
            )
            if not isinstance(current_datasets, list):
                current_datasets = []
        except Exception:
            current_datasets = []

        updated_models = _merge_unique(current_models, models_list)
        updated_datasets = _merge_unique(current_datasets, datasets_list)

        mapping: dict[str, str] = {}
        if updated_models != current_models:
            mapping["cache_models_json"] = json.dumps(
                updated_models, ensure_ascii=False
            )
        if updated_datasets != current_datasets:
            mapping["cache_datasets_json"] = json.dumps(
                updated_datasets, ensure_ascii=False
            )
        if mapping:
            mapping["cache_updated_ts"] = now_iso()
        if not mapping:
            return

        try:
            self._rds.sync.hash_set(worker_key(worker_id), mapping)
        except Exception:
            return

    def idle_satisfying_pool(self, task: TaskEnvelope) -> list[Worker]:
        available: list[Worker] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker or worker.status is not WorkerStatus.IDLE:
                continue
            if self.is_worker_stale(worker.id):
                continue
            if hw_satisfies(worker, task) and capability_satisfies(worker, task):
                available.append(worker)
        return self.sort_workers(available)

    def satisfying_workers(self, task: TaskEnvelope) -> list[Worker]:
        """Non-stale workers whose hardware and capabilities satisfy the task."""
        available: list[Worker] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker or self.is_worker_stale(worker.id):
                continue
            if hw_satisfies(worker, task) and capability_satisfies(worker, task):
                available.append(worker)
        return self.sort_workers(available)

    def sort_workers(self, workers: list[Worker]) -> list[Worker]:
        decorated: list[tuple[Worker, int, int, int, int]] = []
        for worker in workers:
            hardware = worker.hardware
            gpu_entries = [] if hardware is None else hardware.gpu.devices
            gpu_count = len(gpu_entries)
            total_vram = dedicated_gpu_memory_total_bytes(hardware)
            sys_ram = 0 if hardware is None else (hardware.memory.total_bytes or 0)
            cpu_cores = 0 if hardware is None else hardware.cpu.logical_cores
            decorated.append((worker, gpu_count, total_vram, sys_ram, cpu_cores))

        decorated.sort(key=lambda item: item[0].id)
        decorated.sort(
            key=lambda item: (
                item[1] > 0,
                item[2],
                item[3],
                item[4],
            ),
            reverse=True,
        )
        return [item[0] for item in decorated]

    def publish_task(self, worker: Worker, payload: WorkerTaskMessage) -> int:
        payload_json = payload.model_dump(mode="json", exclude_none=True, by_alias=True)
        message = TaskMessage(
            worker_id=worker.id,
            payload=payload_json,
        ).model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_task_async(
        self, worker: Worker, payload: WorkerTaskMessage
    ) -> int:
        payload_json = payload.model_dump(mode="json", exclude_none=True, by_alias=True)
        message = TaskMessage(
            worker_id=worker.id,
            payload=payload_json,
        ).model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_interrupt(self, worker: Worker, payload: InterruptMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_interrupt_async(
        self, worker: Worker, payload: InterruptMessage
    ) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_stop(self, worker: Worker, payload: StopMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_stop_async(self, worker: Worker, payload: StopMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def _allocate_worker_id(self) -> str:
        seq = self._rds.sync.incr(WORKER_ID_SEQ_KEY)
        return new_worker_id(seq)

    async def _allocate_worker_id_async(self) -> str:
        seq = await self._rds.asyncio.incr(WORKER_ID_SEQ_KEY)
        return new_worker_id(seq)


# --- Helper functions --- #


def _normalize_cache_value(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _merge_unique(existing: list[str], additions: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in existing:
        normalized = _normalize_cache_value(value)
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        merged.append(normalized)
    for value in additions:
        normalized = _normalize_cache_value(value)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged


def hw_satisfies(worker: Worker, task: TaskEnvelope) -> bool:
    resources = task.spec.resources
    if resources is None:
        return True
    requirements = resources.hardware
    if requirements is None:
        return True

    hw = worker.hardware
    cpu_needed = requirements.cpu
    mem_needed = requirements.memory
    gpu_req = requirements.gpu

    # Consider SSH hardware limits for SSH tasks.
    ssh_caps = (
        worker.ssh_limits
        if isinstance(task.spec, (SSHSpecStrict, SSHSpecTemplate))
        and worker.ssh_limits is not None
        else None
    )

    if cpu_needed is not None:
        cpu_cores: float | None = None if hw is None else hw.cpu.logical_cores
        if (
            ssh_caps is not None
            and ssh_caps.max_cpu_cores is not None
            and cpu_cores is not None
        ):
            cpu_cores = min(cpu_cores, ssh_caps.max_cpu_cores)
        if cpu_cores is None or cpu_cores < cpu_needed:
            return False

    if mem_needed:
        required_bytes = parse_mem_to_bytes(str(mem_needed)) or 0
        available = 0 if hw is None else (hw.memory.total_bytes or 0)
        if (
            ssh_caps is not None
            and ssh_caps.max_memory_bytes is not None
            and available > 0
        ):
            available = min(available, ssh_caps.max_memory_bytes)
        if available < required_bytes:
            return False

    if gpu_req:
        if hw is None:
            return False
        if not _gpu_meets_requirements(hw, gpu_req):
            return False

    return True


def capability_satisfies(worker: Worker, task: TaskEnvelope) -> bool:
    return task.spec.taskType in worker.capabilities.supported_task_types


def _gpu_meets_requirements(hw: WorkerHardware, gpu_req: GPURequirements) -> bool:
    required_count = gpu_req.count
    if required_count is not None:
        try:
            required_count = int(required_count)
        except Exception:
            required_count = None
    required_type = normalize_gpu_type(gpu_req.type)
    required_memory_bytes = parse_gpu_memory_bytes(gpu_req.memory)
    needed = required_count or 1

    entries = hw.gpu.devices
    if entries:
        if required_count is not None and len(entries) < required_count:
            return False
        if len(select_matching_gpu_indices(entries, gpu_req)) >= needed:
            return True
        # Unified-memory fallback: when memory is the binding constraint and
        # the worker exposes a unified GPU/system pool large enough to cover
        # the request, still admit it.
        if required_memory_bytes is None:
            return False
        type_only_req = GPURequirements(
            count=gpu_req.count, type=gpu_req.type, memory=None
        )
        if len(select_matching_gpu_indices(entries, type_only_req)) < needed:
            return False
        return unified_gpu_memory_satisfies(hw, required_memory_bytes, needed)

    # Fallback when workers report aggregate GPU data instead of per-device entries.
    count = 0 if hw is None else len(hw.gpu.devices)
    if required_count is not None and count < required_count:
        return False

    first_gpu = hw.gpu.devices[0] if hw and hw.gpu.devices else None
    type_value = first_gpu.name if first_gpu else None
    if required_type and not str(type_value or "").strip().lower().startswith(
        required_type
    ):
        return False

    if required_memory_bytes:
        total_mem = 0 if first_gpu is None else (first_gpu.memory_total_bytes or 0)
        if total_mem <= 0 and unified_gpu_memory_satisfies(
            hw, required_memory_bytes, needed
        ):
            return True
        if total_mem <= 0:
            return False
        per_gpu = total_mem / max(needed, 1)
        if per_gpu < required_memory_bytes:
            return False

    return True


def dedicated_gpu_memory_total_bytes(hw: WorkerHardware | None) -> int:
    if hw is None:
        return 0
    total = 0
    for entry in hw.gpu.devices:
        total += entry.memory_total_bytes or 0
    return total


def _parse_worker_from_redis(
    worker_id: str, value: dict[str, Any] | None
) -> Worker | None:
    if not value:
        return None

    def _loads(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def _ensure_str_list(items: Any) -> list[str]:
        if not isinstance(items, list):
            return []
        result: list[str] = []
        for item in items:
            if isinstance(item, str):
                norm = item.strip()
                if norm:
                    result.append(norm)
        return result

    env = _loads(value.get("env_json"), {})
    hardware_json = value.get("hardware_json")
    hardware = (
        None
        if hardware_json is None
        else WorkerHardware.model_validate_json(hardware_json)
    )
    capabilities_json = value.get("capabilities_json")
    capabilities = (
        WorkerCapabilities()
        if capabilities_json is None
        else WorkerCapabilities.model_validate_json(capabilities_json)
    )
    ssh_limits_json = value.get("ssh_limits_json")
    ssh_limits = (
        None
        if ssh_limits_json is None
        else SSHLimits.model_validate_json(ssh_limits_json)
    )
    tags = _loads(value.get("tags_json"), [])
    cached_models = _ensure_str_list(_loads(value.get("cache_models_json"), []))
    cached_datasets = _ensure_str_list(_loads(value.get("cache_datasets_json"), []))
    cache_updated_ts = value.get("cache_updated_ts") or None

    pid_val = value.get("pid")
    try:
        pid = int(pid_val) if pid_val is not None else None
    except (TypeError, ValueError):
        pid = None

    cost_val = value.get("cost_per_hour")
    try:
        cost_per_hour = float(cost_val) if cost_val is not None else None
    except (TypeError, ValueError):
        cost_per_hour = None

    return Worker(
        id=value.get("id", worker_id),
        alias=value.get("alias"),
        namespace=value.get("namespace", ""),
        cluster=value.get("cluster", ""),
        node_id=value.get("node_id", ""),
        node_alias=value.get("node_alias", ""),
        version=value.get("version"),
        status=WorkerStatus(value.get("status", "UNKNOWN")),
        started_at=value.get("started_at"),
        pid=pid,
        env=env,
        hardware=hardware,
        capabilities=capabilities,
        ssh_limits=ssh_limits,
        tags=tags,
        last_seen=value.get("last_seen"),
        cached_models=cached_models,
        cached_datasets=cached_datasets,
        cache_updated_ts=cache_updated_ts,
        cost_per_hour=cost_per_hour,
    )
