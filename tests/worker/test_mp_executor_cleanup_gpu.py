import logging
import tempfile
import time
import uuid
from pathlib import Path

import psutil
import pynvml  # type: ignore
import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config
from worker.executors.mp_executor import MPExecutor
from worker.executors.vllm_executor import VLLMExecutor

pynvml.nvmlInit()


def _descendants_of(pid: int) -> set[int]:
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return set()
    return {c.pid for c in p.children(recursive=True)}


@pytest.mark.gpu
def test_mp_executor_cleans_up_vllm(caplog, tmp_path: Path) -> None:
    """Start MPExecutor with the real executors, run a minimal task to
    trigger engine startup, and ensure cleanup removes the worker process
    and any descendants it spawned.
    """
    # Capture all logs at DEBUG level to see executor initialization
    caplog.set_level(logging.DEBUG)

    def total_gpu_used() -> int:
        # Sum used memory across all devices (bytes)
        count = pynvml.nvmlDeviceGetCount()
        total = 0
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            total += info.used
        return total

    before = set(psutil.pids())

    gpu_before = total_gpu_used()

    mp = MPExecutor(VLLMExecutor, config=make_live_worker_config(tmp_path))

    # Create task payload matching parse_task_yaml output format
    task_payload = WorkerTaskMessage.model_validate(
        {
            "task_id": str(uuid.uuid4()),
            "workflow_id": "test-workflow",
            "owner_id": "test-owner",
            "assigned_worker": "test-worker",
            "dispatched_at": "2026-03-01T00:00:00Z",
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "InferenceTask",
                "spec": {
                    "taskType": "inference",
                    "resources": {
                        "replicas": 1,
                        "hardware": {
                            "cpu": "8",
                            "memory": "32Gi",
                            "gpu": {"type": "any", "count": 1},
                        },
                    },
                    "model": {
                        "source": {
                            "type": "huggingface",
                            "identifier": "Qwen/Qwen2.5-0.5B-Instruct",
                            "revision": "main",
                        },
                        "vllm": {
                            "tensor_parallel_size": 1,
                            "gpu_memory_utilization": 0.9,
                            "trust_remote_code": True,
                        },
                    },
                    "data": {
                        "type": "dataset",
                        "url": "openai/gsm8k",
                        "name": "main",
                        "split": "train[:1%]",
                        "column": "question",
                        "shuffle": True,
                        "seed": 42,
                    },
                    "inference": {"max_tokens": 128, "temperature": 0.7, "top_p": 0.95},
                },
            },
        }
    )

    mp.run(task_payload, Path(tempfile.mkdtemp(prefix="test-cleanup-")))

    # Let the worker process and potential children settle
    time.sleep(2.0)

    assert mp._proc is not None and mp._proc.pid is not None
    worker_pid = mp._proc.pid

    after = set(psutil.pids())
    new_pids = after - before
    # Worker process should be one of the new pids
    assert worker_pid in new_pids

    # Snapshot descendants spawned by the worker (if any)
    desc = _descendants_of(worker_pid)

    # Now request cleanup/teardown
    mp.cleanup_after_run()

    # Check GPU memory freed (if there was any increase)
    gpu_after = total_gpu_used()
    # If worker didn't allocate GPU memory, skip strict GPU assertions
    if gpu_after <= gpu_before + 1024 * 1024:  # <=1MiB tolerance
        # No discernible GPU allocation detected; skip detailed check
        return

    # Otherwise wait for GPU memory to return near baseline
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if total_gpu_used() <= gpu_before + 10 * 1024 * 1024:  # 10 MiB tolerance
            break
        time.sleep(0.5)

    final_gpu = total_gpu_used()
    assert final_gpu <= gpu_before + 10 * 1024 * 1024, (
        f"GPU memory not released after cleanup: "
        f"before={gpu_before}, after_cleanup={final_gpu}"
    )

    # Wait up to 15s for worker and descendants to disappear
    deadline = time.time() + 15.0
    while time.time() < deadline:
        still_alive = [p for p in [worker_pid, *desc] if psutil.pid_exists(p)]
        if not still_alive:
            break
        time.sleep(0.5)

    leftover = [p for p in [worker_pid, *desc] if psutil.pid_exists(p)]
    assert (
        not leftover
    ), f"Leftover PIDs after cleanup: " f"{leftover} - cmdlines: " + ", ".join(
        (
            f"{p!r}:{psutil.Process(p).cmdline()}"
            if psutil.pid_exists(p)
            else f"{p!r}:<gone>"
        )
        for p in leftover
    )

    # Also ensure MPExecutor's process object is not alive
    if mp._proc is not None:
        assert not mp._proc.is_alive()
