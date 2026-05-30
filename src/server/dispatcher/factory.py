import logging
from pathlib import Path

from ..config import DispatchConfig
from ..dispatcher import Dispatcher
from ..registries.worker import WorkerRegistry
from ..services.metrics import MetricsRecorder
from ..task.runtime import TaskRuntime

DEFAULT_DISPATCH_MODE = "adaptive"

_DISPATCHER_REGISTRY: dict[str, type[Dispatcher]] = {
    "adaptive": Dispatcher,
}


def create_dispatcher(
    config: DispatchConfig,
    runtime: TaskRuntime,
    worker_registry: WorkerRegistry,
    results_dir: Path,
    logger: logging.Logger,
    metrics_recorder: MetricsRecorder | None = None,
) -> Dispatcher:
    """
    Instantiate a dispatcher according to the selected mode.

    Supported modes:
    - adaptive (default): capability-aware dispatcher
    """

    normalized = (config.mode or DEFAULT_DISPATCH_MODE).strip().lower()
    if normalized not in _DISPATCHER_REGISTRY:
        logger.warning(
            "Unknown dispatcher mode '%s'; falling back to adaptive dispatcher",
            config.mode or "<empty>",
        )
        normalized = DEFAULT_DISPATCH_MODE

    dispatcher_cls: type[Dispatcher] = _DISPATCHER_REGISTRY[normalized]
    logger.info("Initializing dispatcher in %s mode", normalized)
    lambda_config = {
        "inference": config.lambda_inference,
        "training": config.lambda_training,
        "other": config.lambda_other,
    }
    return dispatcher_cls(
        runtime=runtime,
        worker_registry=worker_registry,
        results_dir=results_dir,
        logger=logger,
        worker_selection_strategy=config.worker_selection,
        enable_context_reuse=config.enable_context_reuse,
        enable_task_merge=config.enable_task_merge,
        task_merge_max_batch_size=config.task_merge_max_batch_size,
        reuse_cache_ttl_sec=config.worker_cache_ttl_sec,
        lambda_config=lambda_config,
        selection_jitter_epsilon=config.selection_jitter,
        enable_stage_weight_stickiness=config.enable_stage_weight_stickiness,
        no_worker_grace_sec=config.no_worker_grace_sec,
        metrics_recorder=metrics_recorder,
    )
