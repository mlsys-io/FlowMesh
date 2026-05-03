"""In-process replacements for the ``torchrun`` and ``deepspeed`` launchers.

The training executors (SFT / DPO / PPO) launch multi-GPU runs by re-spawning
themselves through ``torchrun -m worker.executors.<X>_dist_entry ...`` or
``deepspeed --num_gpus N --module ...``. Going through either CLI requires
``import subprocess`` (B404) at the executor module level and an implicit
dependency on the binary being on ``$PATH``.

``torch.distributed.run.main`` and ``deepspeed.launcher.runner.main`` are the
*canonical* entry points the two CLIs invoke — the ``torchrun`` console script
in PyTorch is registered as ``torch.distributed.run:main``, and the
``deepspeed`` console script is literally
``from deepspeed.launcher.runner import main; main()``. Calling them in-process
is the documented in-process equivalent; worker ranks are still spawned by
torch's elastic agent / DeepSpeed's launcher under the hood, so the runtime
semantics are unchanged.
"""

import importlib.util
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


def _launch_env(launcher_env_flag: str) -> dict[str, str]:
    pythonpath = _REPO_ROOT.as_posix()
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    return {"PYTHONPATH": pythonpath, launcher_env_flag: "1"}


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

    ``--tee 3`` (stdout+stderr bitmask) keeps the rank streams on the
    parent's console *and* writes per-rank log files under the elastic
    agent's log dir. Without it the elastic agent silently swallows rank
    output and a rank-side crash is reported as an opaque
    ``ChildFailedError`` with ``error_file: <N/A>``.
    """
    with _scoped_env(_launch_env(launcher_env_flag)):
        _torchrun_main(
            [
                "--nproc_per_node",
                str(nproc_per_node),
                "--tee",
                "3",
                "-m",
                module,
                *module_args,
            ]
        )


def deepspeed_available() -> bool:
    """Whether ``deepspeed.launcher.runner`` is importable in this environment.

    The CPU worker image does not ship DeepSpeed (it is a ``training-gpu``
    extra), so callers must guard ``run_deepspeed`` with this check the same
    way the previous implementation guarded ``shutil.which("deepspeed")``.

    Any exception raised while resolving the spec is treated as "not
    available" — DeepSpeed's package init eagerly probes CUDA op builders, so
    a CUDA-less host (e.g. a CPU CI runner with the GPU extra installed)
    raises ``MissingCUDAException`` here. Returning ``False`` in that case is
    the right answer: ``run_deepspeed`` would also fail on the same import.
    """
    try:
        return importlib.util.find_spec("deepspeed.launcher.runner") is not None
    except Exception:
        return False


def run_deepspeed(
    *,
    num_gpus: int,
    module: str,
    module_args: list[str],
    launcher_env_flag: str,
) -> None:
    """Run ``deepspeed --num_gpus N --module <module> <args>`` in-process.

    Same env-scoping contract as :func:`run_torchrun`. Imported lazily so the
    CPU worker image (which does not install DeepSpeed) does not pay the
    import cost or fail at module load time.
    """
    from deepspeed.launcher.runner import main as _deepspeed_main

    with _scoped_env(_launch_env(launcher_env_flag)):
        _deepspeed_main(
            [
                "--num_gpus",
                str(num_gpus),
                "--module",
                module,
                *module_args,
            ]
        )
