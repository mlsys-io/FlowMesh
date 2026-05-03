"""Distributed launcher helpers used by the training executors.

Exposes :func:`run_torchrun` and :func:`run_deepspeed`, which invoke
``torch.distributed.run.main`` and ``deepspeed.launcher.runner.main`` directly
— the same entry points the ``torchrun`` and ``deepspeed`` console scripts
call. Worker ranks are spawned by torch's elastic agent / DeepSpeed's
launcher; this module just shapes the argv and scopes the launcher env.

:func:`deepspeed_available` reports whether the DeepSpeed package can be
imported in the current environment, so callers can fall back to torchrun
when DeepSpeed is absent (CPU worker image) or unusable (no CUDA toolchain).
"""

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from torch.distributed.run import main as _torchrun_main

# .../src/worker/executors/utils/distributed.py → parents[3] = .../src
_SRC_DIR = Path(__file__).resolve().parents[3]


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
    pythonpath = _SRC_DIR.as_posix()
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

    The launcher prepends the source root to ``PYTHONPATH`` so spawned ranks
    can import ``worker.executors.*``, and sets ``launcher_env_flag`` to
    ``"1"`` so the entry module can detect it is running inside the launched
    ranks and skip a second spawn. Both env mutations are scoped to the
    launch call — the caller's environment is restored on return and on
    exception.

    ``--tee 3`` (stdout+stderr bitmask) is passed so the rank streams reach
    the parent's console and per-rank log files are written under the
    elastic agent's log dir; without it the elastic agent swallows rank
    output and a rank-side crash surfaces as an opaque ``ChildFailedError``
    with ``error_file: <N/A>``.
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
    """Return whether ``deepspeed.launcher.runner`` is importable here.

    Use this to guard :func:`run_deepspeed` calls — DeepSpeed is a
    ``training-gpu`` extra and is absent from the CPU worker image. Any
    exception raised while resolving the spec is treated as "not available";
    DeepSpeed's package init eagerly probes CUDA op builders and raises
    ``MissingCUDAException`` on a CUDA-less host, which is indistinguishable
    from "not usable here".
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

    The launcher prepends the source root to ``PYTHONPATH`` so spawned ranks
    can import ``worker.executors.*``, and sets ``launcher_env_flag`` to
    ``"1"`` so the entry module can detect it is running inside the launched
    ranks. Both env mutations are scoped to the launch call — the caller's
    environment is restored on return and on exception. The DeepSpeed
    launcher is imported lazily so the CPU worker image (which omits the
    DeepSpeed dependency) is unaffected.
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
