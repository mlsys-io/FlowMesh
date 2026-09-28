"""Executor for ``python`` tasks: the caller's function in its own container.

A python task is a hardened non-interactive session on the Docker session
backend: the same per-task container, cgroup limits, upstream-input mounts and
output collection as an SSH task (``SessionExecutor``), with these differences
set on the resolved config:

* no network (``network: none``, the default) — or the isolated SSH bridge
  when the spec asks for ``bridge`` (needed to pip-install ``requirements``);
* all capabilities dropped but the few the bootstrap needs before it switches
  to an unprivileged uid, with ``/tmp``, ``/var/tmp`` and ``/run/lock`` as
  in-memory scratch (anything else it writes is bounded by ``SSH_MAX_DISK``
  when set);
* the caller's code and the bootstrap arrive as files, not environment;
* the task succeeds only when its process exits 0: a timeout, a finish request
  or a lost container is a failure.

There is deliberately no process-backend fallback: on a worker without Docker
the executor reports itself unavailable, so the scheduler never places a python
task where it would run unisolated on the host.
"""

import json
import logging
import math
import os
import stat
from pathlib import Path
from typing import Any

from shared.schemas.result import PythonResult
from shared.tasks.specs.python import (
    DEFAULT_TIMEOUT_SECONDS,
    OUTPUT_MOUNT_PATH,
    PythonSpecStrict,
)
from shared.tasks.specs.ssh import SSHInputSpec, SSHSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.manifest import ARTIFACTS_DIR
from worker.config import WorkerConfig
from worker.executors.ssh_session import (
    DockerSessionBackend,
    SSHConfig,
    normalize_mount_path,
    select_backend_cls,
)
from worker.executors.ssh_session.config import DEFAULT_INPUTS_ROOT

from .base_executor import ExecutionError, ExecutorTask, RunControl
from .session_executor import SessionExecutor, SessionOutcome

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.12-slim"
BOOTSTRAP_PATH = "/opt/flowmesh/python-run.py"
CODE_PATH = "/opt/flowmesh/flowmesh_task.py"
UNPRIVILEGED_UID = 65534  # nobody
_BOOTSTRAP_SOURCE = Path(__file__).resolve().parents[1] / "docker" / "python-run.py"


class PythonExecutor(SessionExecutor):
    name = "python"
    supported_task_types = frozenset({TaskType.PYTHON})

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        backend = select_backend_cls(config)
        return backend is not None and issubclass(backend, DockerSessionBackend)

    def run(
        self, task: ExecutorTask, out_dir: Path, control: RunControl
    ) -> PythonResult:
        spec = self.require_spec(task, PythonSpecStrict)
        if spec.inputs is None and task.upstream_task_ids:
            # No explicit inputs: the dispatcher resolved every direct
            # dependency, and each one is mounted under its stage name.
            spec = spec.model_copy(
                update={
                    "inputs": [
                        SSHInputSpec(stage=stage) for stage in task.upstream_task_ids
                    ]
                }
            )
        cfg = self._python_config(spec)
        outcome = self._run_session(task, out_dir, cfg, control)
        artifacts = out_dir / ARTIFACTS_DIR
        _raise_unless_succeeded(outcome, cfg.ttl_sec, artifacts)
        return _read_result(artifacts, spec.emits or [])

    def _python_config(self, spec: PythonSpecStrict) -> SSHConfig:
        inputs = {
            entry.stage.strip(): normalize_mount_path(
                entry.mountPath or f"{DEFAULT_INPUTS_ROOT}/{entry.stage.strip()}",
                field_name=f"inputs[{entry.stage}].mountPath",
            )
            for entry in spec.inputs or []
        }
        env: dict[str, Any] = {
            **(spec.env or {}),
            "FLOWMESH_PY_CODE": CODE_PATH,
            "FLOWMESH_PY_ENTRYPOINT": spec.entrypoint,
            "FLOWMESH_PY_INPUTS": json.dumps(inputs),
            "FLOWMESH_PY_OUTPUT": OUTPUT_MOUNT_PATH,
            "FLOWMESH_PY_REQUIREMENTS": json.dumps(spec.requirements or []),
            "FLOWMESH_PY_EMITS": json.dumps(spec.emits or []),
            "FLOWMESH_PY_UID": str(UNPRIVILEGED_UID),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if not spec.uses_gpu():
            # Also hides the cards on a host whose default runtime is nvidia.
            env["NVIDIA_VISIBLE_DEVICES"] = "void"
        # Resolve through the SSH spec so resource caps, the worker's ssh_limits
        # and GPU selection apply exactly as they do to any other session.
        session_spec = SSHSpecStrict.model_validate(
            {
                "taskType": TaskType.SSH,
                "interactive": False,
                "image": spec.image or DEFAULT_IMAGE,
                "command": ["python3", BOOTSTRAP_PATH],
                "ttlSeconds": spec.timeoutSeconds or DEFAULT_TIMEOUT_SECONDS,
                "inputs": [i.model_dump() for i in spec.inputs or []],
                "sshOutput": {
                    "mountPath": OUTPUT_MOUNT_PATH,
                    "maxBytes": (
                        spec.pythonOutput.maxBytes if spec.pythonOutput else None
                    ),
                },
                "env": env,
                "resources": (
                    spec.resources.model_dump(exclude_none=True)
                    if spec.resources
                    else None
                ),
                "dependsOn": spec.dependsOn,
            }
        )
        cfg = self._config_for(session_spec)
        if not spec.uses_gpu():
            # A session with no gpu block is otherwise handed every host GPU.
            cfg.gpu_device_ids = []
        cfg.network_disabled = spec.network == "none"
        cfg.hardened = True
        # A python task ends only with its result: the finish helper, which the
        # code could reach by touching the sentinel in /tmp, never ends it early.
        cfg.honor_finish_request = False
        cfg.extra_files = {
            BOOTSTRAP_PATH: _BOOTSTRAP_SOURCE.read_bytes(),
            CODE_PATH: spec.code.encode(),
        }
        return cfg


def _read_task_file(path: Path) -> str:
    """The text of a file the task wrote, which must be a regular file.

    Refusing symlinks keeps the worker from reading its own filesystem on the
    task's behalf, and ``O_NONBLOCK`` keeps a FIFO from blocking the open.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError(f"{path.name} is not a regular file")
    with os.fdopen(fd, encoding="utf-8") as fh:
        return fh.read()


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is out of range")
    return value


def _load_task_json(path: Path, finite: bool = False) -> Any:
    """Parse a regular file the task wrote; ``finite`` rejects NaN and infinities."""
    strict: dict[str, Any] = (
        {"parse_constant": _reject_constant, "parse_float": _finite_float}
        if finite
        else {}
    )
    try:
        return json.loads(_read_task_file(path), **strict)
    except (OSError, ValueError) as exc:
        raise ExecutionError(
            f"python task wrote an unreadable {path.name}: {exc}"
        ) from exc


def _read_result(artifacts: Path, emits: list[str]) -> PythonResult:
    """The result of a clean exit, held to the same contract the bootstrap checks.

    The caller's code can end the process itself (``os._exit(0)``) before the
    bootstrap writes or checks anything, so a clean exit alone proves nothing.
    """
    result_file = artifacts / "result.json"
    if not os.path.lexists(result_file):
        raise ExecutionError("python task exited without writing a result")
    value = _load_task_json(result_file)
    metrics_file = artifacts / "metrics.json"
    metrics = (
        _load_task_json(metrics_file, finite=True)
        if os.path.lexists(metrics_file)
        else {}
    )
    if not isinstance(metrics, dict) or not all(
        _is_finite_number(v) for v in metrics.values()
    ):
        raise ExecutionError("python task wrote metrics that are not finite numbers")
    if missing := [name for name in emits if name not in metrics]:
        raise ExecutionError(f"python task did not report declared emits {missing}")
    return PythonResult(exit_code=0, value=value, metrics=metrics)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _raise_unless_succeeded(
    outcome: SessionOutcome, timeout_sec: float, artifacts: Path
) -> None:
    """A python task succeeds only when its process exits 0 on its own."""
    end = outcome.end
    match end.reason:
        case "exited" if end.exit_code == 0:
            return
        case "exited" if end.exit_code == 137:
            raise ExecutionError(
                "python task was killed (exit 137), most often by its memory limit"
            )
        case "exited":
            raise ExecutionError(_failure_message(artifacts, end.exit_code))
        case "ttl":
            raise ExecutionError(f"python task timed out after {timeout_sec:g}s")
        case "lost":
            raise ExecutionError("python task's container was lost before it exited")
        case _:
            raise ExecutionError("python task was stopped before it finished")


def _failure_message(artifacts: Path, exit_code: int) -> str:
    """The caller's own error (from error.json) beats the bare exit code."""
    error_file = artifacts / "error.json"
    error = None
    if os.path.lexists(error_file):
        try:
            error = _load_task_json(error_file)
        except ExecutionError as exc:
            logger.warning("%s", exc)
    if isinstance(error, dict) and error.get("message") is not None:
        return f"python task failed: {error.get('type')}: {error['message']}"
    return f"python task exited with code {exit_code}"
