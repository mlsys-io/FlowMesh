"""Regression tests for executor bootstrap in ``worker.main.initialize_executors``.

Pins the contract that ``initialize_executors`` constructs every non-MP executor
with ``cls(config, hardware, lifecycle)``. Subclasses are expected to accept this
via ``(*args, **kwargs)`` passthrough so future ``Executor.__init__`` extensions
don't break the chain.
"""

import logging
from pathlib import Path
from typing import Any

from shared.tasks.worker_message import (
    CPUInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from tests.worker.factories import make_live_worker_config
from worker.executors.base_executor import Executor, ExecutorTask
from worker.main import initialize_executors


class _PassthroughExecutor(Executor):
    """Executor that forwards constructor args via the recommended pattern."""

    name = "passthrough"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:  # noqa: D401
        return {"ok": True}


def _hardware() -> WorkerHardware:
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=2, model="x"),
        memory=MemoryInfo(total_bytes=1024**3),
        gpu=GpuPlatformInfo(
            driver_version=None,
            cuda_version=None,
            devices=[],
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


class TestInitializeExecutorsHardware:
    def test_executor_receives_hardware_via_passthrough(self, tmp_path: Path) -> None:
        cfg = make_live_worker_config(tmp_path)
        hw = _hardware()
        executors, default = initialize_executors(
            config=cfg,
            hardware=hw,
            logger=logging.getLogger("test"),
            lifecycle=None,  # type: ignore[arg-type]
            registry={"echo": _PassthroughExecutor, "default": _PassthroughExecutor},
            import_errors={},
            cuda_available=False,
            enable_mp_executors=False,
        )
        # Pre-fix this would silently drop the executor because the subclass
        # constructor didn't accept the new positional ``hardware`` arg.
        assert isinstance(executors["echo"], _PassthroughExecutor)
        assert isinstance(default, _PassthroughExecutor)
        assert executors["echo"]._hardware is hw
        assert default._hardware is hw
