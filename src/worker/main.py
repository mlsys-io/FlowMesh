import argparse
import logging
import signal
from collections.abc import Mapping

from shared.schemas.worker import WorkerCapabilities
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerHardware

from .config import WorkerConfig
from .executors import EXECUTOR_REGISTRY, IMPORT_ERRORS, get_executor_class_name
from .executors.base_executor import Executor
from .executors.mp_executor import MPExecutor
from .hw import collect_hw
from .lifecycle import Lifecycle
from .power import PowerMonitor
from .runner import Runner
from .supervisor_client import SupervisorClient
from .utils.logging import get_logger

_EXECUTORS_TO_WRAP = {
    "default",
    "vllm",
    "vllm_lora",
    "vllm_embedding",
    "sft",
    "lora_sft",
    "image_classification_training",
    "ppo",
    "dpo",
    "data_profiling",
    "data_retrieval",
    "diffusers",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowMesh worker entrypoint.")
    parser.add_argument(
        "--collect-hw",
        action="store_true",
        help="Print hardware JSON and exit.",
    )
    parser.add_argument(
        "--collect-hw-prefix",
        type=str,
        default=None,
        help="Optional prefix to print before hardware JSON.",
    )
    parser.add_argument(
        "--bandwidth-bytes-per-sec",
        type=float,
        default=None,
        help="Optional network bandwidth in bytes per second.",
    )
    return parser.parse_args()


def initialize_executors(
    config: WorkerConfig,
    hardware: WorkerHardware,
    logger: logging.Logger,
    lifecycle: Lifecycle,
    registry: Mapping[str, type[Executor] | None] | None = None,
    import_errors: dict[str, str] | None = None,
    cuda_available: bool | None = None,
    enable_mp_executors: bool = True,
):
    """Initialize executor registry and handle graceful degradation.

    Allows dependency/GPU overrides in tests; returns (executors, default_executor).
    """

    registry = registry or EXECUTOR_REGISTRY
    import_errors = import_errors or IMPORT_ERRORS

    def check_cuda() -> bool:
        if cuda_available is not None:
            return cuda_available
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    configured_wrapped = _EXECUTORS_TO_WRAP if enable_mp_executors else set()

    def init_executor(key: str, *, gpu_required: bool = False):
        cls = registry.get(key)
        if cls is None:
            reason = import_errors.get(
                get_executor_class_name(key, key), "dependency missing"
            )
            logger.info("Skipping executor %s: %s", key, reason)
            return None

        if gpu_required and not check_cuda():
            logger.info("Executor %s requires a GPU; unavailable, skipping", key)
            return None

        if not cls.is_available(config):
            logger.info("Executor %s is unavailable; skipping", key)
            return None

        try:
            if key in configured_wrapped:
                return MPExecutor(cls, config, hardware)
            return cls(config, hardware, lifecycle)
        except Exception as exc:
            logger.warning("Failed to initialize executor %s: %s", key, exc)
            return None

    executors: dict[str, Executor] = {}
    default_executor = init_executor("default")
    if default_executor:
        executors["default"] = default_executor

    for key in [
        "echo",
        "rag",
        "agent",
        "sft",
        "lora_sft",
        "image_classification_training",
        "data_profiling",
        "data_retrieval",
        "diffusers",
        "api",
        "ssh",
    ]:
        inst = init_executor(key)
        if inst:
            executors[key] = inst

    for key in [
        "vllm",
        "vllm_lora",
        "vllm_embedding",
        "vllm_serve",
        "ppo",
        "dpo",
        "omni_text2image",
        "omni_text2speech",
        "omni_text2audio",
        "omni_text2general",
    ]:
        inst = init_executor(key, gpu_required=True)
        if inst:
            executors[key] = inst

    if not executors:
        raise SystemExit(
            "No executors available. Install at least one executor package."
        )

    if not default_executor:
        default_executor = executors.get("echo") or executors.get("api")
        if default_executor is None:
            raise SystemExit(
                "No suitable default executor available. "
                "Ensure the echo/api executor can be initialized."
            )
        logger.info(
            "HFTransformers unavailable; using %s as default executor (CPU-only mode)",
            type(default_executor).__name__,
        )

    return executors, default_executor


def build_capabilities(
    executors: dict[str, Executor],
    registry: Mapping[str, type[Executor] | None] | None = None,
) -> WorkerCapabilities:
    registry = registry or EXECUTOR_REGISTRY
    supported_task_types = frozenset[TaskType]().union(
        *(cls.supported_task_types for key in executors if (cls := registry.get(key)))
    )
    return WorkerCapabilities(supported_task_types=supported_task_types)


def main() -> None:
    args = _parse_args()
    if args.collect_hw:
        hardware = collect_hw(bandwidth_bytes_per_sec=args.bandwidth_bytes_per_sec)
        hardware_json = hardware.model_dump_json()
        if args.collect_hw_prefix:
            hardware_json = args.collect_hw_prefix + hardware_json
        print(hardware_json)
        return

    cfg = WorkerConfig.from_env()
    logger = get_logger(name="worker", level=cfg.log_level)

    supervisor_client = SupervisorClient(
        worker_token=cfg.worker_token,
        owner_principal=cfg.owner_principal,
        grpc_target=cfg.supervisor_grpc_target,
        worker_namespace=cfg.namespace,
        worker_cluster=cfg.cluster,
        worker_alias=cfg.alias,
        logger=logger,
        grpc_tls_ca_b64=cfg.supervisor_grpc_tls_ca_b64,
        grpc_keepalive_time_ms=cfg.grpc_keepalive_time_ms,
        grpc_keepalive_timeout_ms=cfg.grpc_keepalive_timeout_ms,
    )

    lifecycle = Lifecycle(
        supervisor_client,
        cfg.hb_interval_sec,
        cfg.hb_ttl_sec,
        cfg.hb_file,
        cost_per_hour=cfg.cost_per_hour,
        power_monitor=PowerMonitor(),
    )
    hardware = collect_hw(bandwidth_bytes_per_sec=cfg.network_bandwidth_bytes_per_sec)
    logger.info("Collected hardware info: %s", hardware)

    executors, default_executor = initialize_executors(
        cfg,
        hardware,
        logger,
        lifecycle,
        enable_mp_executors=cfg.enable_mp_executors,
    )

    capabilities = build_capabilities(executors)
    ssh_limits = cfg.ssh_limits
    if TaskType.SSH in capabilities.supported_task_types:
        if ssh_limits is None:
            logger.warning(
                "SSH resource cap not configured; SSH sessions will be able to access "
                "full host resources of this worker."
            )
        else:
            logger.info("SSH resource cap: %s", ssh_limits.model_dump())
    lifecycle.start(
        env={},
        hardware=hardware,
        capabilities=capabilities,
        ssh_limits=ssh_limits,
        tags=cfg.tags,
    )

    task_stream = supervisor_client.iter_tasks()
    runner = Runner(
        lifecycle,
        task_stream,
        cfg.results_dir,
        hardware,
        executors,
        default_executor,
        logger,
        network_bandwidth_bytes_per_sec=cfg.network_bandwidth_bytes_per_sec,
        executor_idle_cleanup_sec=cfg.executor_idle_cleanup_sec,
    )

    # Install signal handlers to allow graceful shutdown
    def handle_exit_signal(signum: int, _) -> None:
        logger.info("Received exit signal %d; initiating shutdown", signum)
        runner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        try:
            signal.signal(sig, handle_exit_signal)
        except (ValueError, OSError):
            # Signal not supported on this platform
            logger.debug("Signal %s not supported; skipping handler installation", sig)

    try:
        runner.start()
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()
