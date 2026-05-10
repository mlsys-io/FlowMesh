import hashlib
import logging
from collections.abc import Iterable
from typing import Any

from ..registries.worker import Worker, dedicated_gpu_memory_total_bytes

DEFAULT_WORKER_SELECTION = "best_fit"


def select_worker(
    pool: Iterable[Worker],
    strategy: str = DEFAULT_WORKER_SELECTION,
    logger: logging.Logger | None = None,
    task_category: str | None = None,
    lambda_overrides: dict[str, float] | None = None,
    task_id: str | None = None,
    jitter_epsilon: float = 1e-3,
    task_age: float | None = None,
) -> tuple[Worker | None, dict[str, Any]]:
    """
    Select a worker from the candidate pool according to the scheduling strategy.

    Strategies:
    - best_fit (default): delegate to sort_workers for capacity-aware ordering.
    - first_fit: accept the first worker in the given iterable without reordering.
    - min_satisfying: pick the smallest-capacity worker that still satisfies the task.
    """

    candidates: list[Worker] = list(pool or [])
    if not candidates:
        return None, {"strategy": strategy, "reason": "empty_pool"}

    normalized = (strategy or DEFAULT_WORKER_SELECTION).strip().lower()

    chosen: Worker | None
    if normalized == "first_fit":
        chosen = candidates[0]
        info = {
            "strategy": "first_fit",
            "candidate_count": len(candidates),
            "chosen": chosen.id,
            "task_age": task_age,
        }
        return chosen, info

    if normalized == "min_satisfying":
        chosen, info = _select_min_capacity(
            candidates,
            task_id=task_id,
            jitter_epsilon=jitter_epsilon,
            task_age=task_age,
        )
        if chosen is None and logger:
            logger.debug(
                "Min-capacity selection returned no worker; pool size=%d",
                len(candidates),
            )
        return chosen, info

    if normalized != "best_fit":
        if logger:
            logger.warning(
                "Unknown worker selection strategy '%s'; falling back to best_fit",
                strategy,
            )
        normalized = DEFAULT_WORKER_SELECTION

    chosen, debug = _select_best_fit(
        candidates,
        task_category=task_category,
        lambda_overrides=lambda_overrides,
        task_id=task_id,
        jitter_epsilon=jitter_epsilon,
        task_age=task_age,
    )
    if chosen is None and logger:
        logger.debug(
            "Best-fit selection returned no worker; pool size=%d", len(candidates)
        )
    return chosen, debug


def _select_best_fit(
    candidates: list[Worker],
    *,
    task_category: str | None,
    lambda_overrides: dict[str, float] | None,
    task_id: str | None,
    jitter_epsilon: float,
    task_age: float | None,
) -> tuple[Worker | None, dict[str, Any]]:
    lambda_config = lambda_overrides or {}
    category_key = (task_category or "other").lower()
    lam = float(lambda_config.get(category_key, lambda_config.get("other", 0.5)))
    lam = max(0.0, min(1.0, lam))
    scores: list[tuple[float, Worker, dict[str, float]]] = []

    metric_payloads: list[tuple[Worker, dict[str, float]]] = []
    for worker in candidates:
        metric_payloads.append((worker, _collect_worker_metrics(worker)))

    if not metric_payloads:
        return None, {
            "strategy": "best_fit",
            "candidate_count": 0,
            "reason": "no_scores",
        }

    throughputs = [payload["throughput"] for _, payload in metric_payloads]
    costs = [payload["cost"] for _, payload in metric_payloads]
    throughput_min = min(throughputs)
    throughput_range = max(throughputs) - throughput_min
    cost_min = min(costs)
    cost_range = max(costs) - cost_min

    for worker, metrics in metric_payloads:
        throughput = metrics["throughput"]
        cost = metrics["cost"]
        norm_throughput = (
            0.0
            if throughput_range <= 0
            else (throughput - throughput_min) / throughput_range
        )
        norm_cost = 0.0 if cost_range <= 0 else (cost - cost_min) / cost_range
        score = lam * norm_throughput - (1.0 - lam) * norm_cost
        if task_age is not None:
            score += min(task_age, 300.0) * 1e-4
        if task_id:
            score += _stable_jitter(task_id, worker.id, jitter_epsilon)
        metrics["normalized_throughput"] = norm_throughput
        metrics["normalized_cost"] = norm_cost
        metrics["lambda"] = lam
        metrics["score"] = score
        scores.append((score, worker, metrics))

    scores.sort(key=lambda item: item[2]["worker_id"])  # stable by worker id
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_worker, best_metrics = scores[0]
    debug = {
        "strategy": "best_fit",
        "candidate_count": len(candidates),
        "chosen": best_worker.id,
        "chosen_metrics": best_metrics,
        "task_age": task_age,
        "top_scores": [
            {
                "worker_id": metrics["worker_id"],
                "score": metrics["score"],
                "throughput": metrics["throughput"],
                "normalized_throughput": metrics.get("normalized_throughput"),
                "cost": metrics["cost"],
                "normalized_cost": metrics.get("normalized_cost"),
            }
            for _, _, metrics in scores[:5]
        ],
    }
    return best_worker, debug


def _select_min_capacity(
    candidates: list[Worker],
    *,
    task_id: str | None,
    jitter_epsilon: float,
    task_age: float | None,
) -> tuple[Worker | None, dict[str, Any]]:
    scored: list[tuple[float, float, str, Worker, dict[str, float]]] = []
    for worker in candidates:
        metrics = _collect_worker_metrics(worker)
        adjusted_throughput = metrics["throughput"]
        if task_id:
            adjusted_throughput += _stable_jitter(task_id, worker.id, jitter_epsilon)
        scored.append(
            (
                adjusted_throughput,
                metrics["cost"],
                worker.id,
                worker,
                metrics,
            )
        )

    if not scored:
        return None, {
            "strategy": "min_satisfying",
            "candidate_count": 0,
            "reason": "no_scores",
        }

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    adjusted, _, _, chosen_worker, chosen_metrics = scored[0]
    chosen_metrics = dict(chosen_metrics)
    chosen_metrics["adjusted_throughput"] = adjusted
    debug = {
        "strategy": "min_satisfying",
        "candidate_count": len(candidates),
        "chosen": chosen_worker.id,
        "chosen_metrics": chosen_metrics,
        "task_age": task_age,
        "top_candidates": [
            {
                "worker_id": entry[4]["worker_id"],
                "throughput": entry[4]["throughput"],
                "adjusted_throughput": entry[0],
                "cost": entry[4]["cost"],
            }
            for entry in scored[:5]
        ],
    }
    return chosen_worker, debug


def _collect_worker_metrics(worker: Worker) -> dict[str, Any]:
    hardware = worker.hardware
    gpus = [] if hardware is None else hardware.gpu.gpus
    gpu_count = len(gpus)
    total_vram = dedicated_gpu_memory_total_bytes(hardware)
    sys_ram = 0 if hardware is None else (hardware.memory.total_bytes or 0)
    cpu_cores = 0 if hardware is None else hardware.cpu.logical_cores
    if hardware is not None and hardware.gpu.memory_is_unified:
        shared_gpu_mem = hardware.gpu.shared_memory_total_bytes or 0
        gpu_mem_score = shared_gpu_mem
        sys_mem_score = 0.0
    else:
        shared_gpu_mem = 0
        gpu_mem_score = total_vram
        sys_mem_score = sys_ram / 2
    throughput = (
        gpu_count * 100.0
        + gpu_mem_score / (1 << 30)
        + sys_mem_score / (1 << 30)
        + cpu_cores * 0.5
    )
    cost = worker.cost_per_hour if worker.cost_per_hour is not None else 1.0

    gb = 1 << 30
    return {
        "worker_id": worker.id,
        "throughput": throughput,
        "cost": cost,
        "gpu_count": float(gpu_count),
        "vram_gb": total_vram / gb,
        "shared_gpu_mem_gb": shared_gpu_mem / gb,
        "cpu_cores": float(cpu_cores),
        "sys_ram_gb": sys_ram / gb,
    }


def _stable_jitter(task_id: str, worker_id: str, magnitude: float) -> float:
    if magnitude <= 0:
        return 0.0
    payload = f"{task_id}:{worker_id}".encode("utf-8", "ignore")
    digest = hashlib.md5(payload, usedforsecurity=False).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return (value - 0.5) * magnitude * 2
