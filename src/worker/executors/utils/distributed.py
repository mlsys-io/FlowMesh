"""In-process replacement for ``subprocess.check_call(["torchrun", ...])``.

The training executors (SFT / DPO / PPO) launch multi-GPU runs by re-spawning
themselves through ``torchrun -m worker.executors.<X>_dist_entry ...``. Going
through the ``torchrun`` CLI requires ``import subprocess`` (B404) at the
executor module level and an implicit dependency on ``torchrun`` being on
``$PATH``. ``torch.distributed.run.main`` is the same entry point ``torchrun``
calls — invoking it directly drops both. Worker ranks are still spawned by
torch's elastic agent under the hood, so the runtime semantics are unchanged.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from torch.distributed.run import main as _torchrun_main

# .../src/worker/executors/utils/distributed.py → parents[3] = .../src
_REPO_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def _scoped_env(updates: dict[str, str]) -> Iterator[None]:
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def run_torchrun(
    *,
    nproc_per_node: int,
    module: str,
    module_args: list[str],
    launcher_env_flag: str,
) -> None:
    """Run ``torchrun --nproc_per_node N -m <module> <args>`` in-process.

    ``PYTHONPATH`` is prefixed with ``src/`` so the spawned ranks can import
    ``worker.executors.*``; ``launcher_env_flag`` is set to ``"1"`` so the
    entry module can detect it is running inside the launched ranks and not
    re-recurse into another launch. Both env mutations are scoped to the
    launch call — the caller's environment is restored on return (and on
    exception), so reusing the executor instance for a second task does not
    see the launcher flag pre-set.
    """
    pythonpath = _REPO_ROOT.as_posix()
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    with _scoped_env({"PYTHONPATH": pythonpath, launcher_env_flag: "1"}):
        _torchrun_main(
            [
                "--nproc_per_node",
                str(nproc_per_node),
                "-m",
                module,
                *module_args,
            ]
        )
