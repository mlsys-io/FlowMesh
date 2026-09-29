"""Executor for ``python`` tasks: the caller's function in its own container.

A python task is a hardened non-interactive session on the Docker session
backend: the same per-task container, cgroup limits, GPU slice, upstream-input
mounts and output collection the SSH executor uses (``_run_session``), with
three differences set on the resolved config:

* no network (``network: none``, the default) — or the isolated SSH bridge
  when the spec asks for ``bridge`` (needed to pip-install ``requirements``);
* all capabilities dropped but the few the bootstrap needs before it switches
  to an unprivileged uid, and ``/tmp`` as the only writable scratch;
* the caller's code and the bootstrap arrive as files, not environment.

There is deliberately no process-backend fallback: on a worker without Docker
the executor reports itself unavailable, so the scheduler never places a python
task where it would run unisolated on the host.
"""

import json
import logging
from pathlib import Path
from typing import Any

from shared.schemas.result import PythonResult
from shared.tasks.specs.python import (
    DEFAULT_TIMEOUT_SECONDS,
    PythonSpecStrict,
)
from shared.tasks.specs.ssh import SSHSpecStrict
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

from .base_executor import ExecutionError, ExecutorTask
from .ssh_executor import SessionExitError, SSHExecutor

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.12-slim"
OUTPUT_PATH = "/mnt/flowmesh/output"
BOOTSTRAP_PATH = "/opt/flowmesh/python-run.py"
CODE_PATH = "/opt/flowmesh/task.py"
UNPRIVILEGED_UID = 65534  # nobody
_BOOTSTRAP_SOURCE = Path(__file__).resolve().parents[1] / "docker" / "python-run.py"


class PythonExecutor(SSHExecutor):
    name = "python"
    supported_task_types = frozenset({TaskType.PYTHON})
    # A python task must end with a result. Running past timeoutSeconds is a
    # failure (124, as timeout(1) reports it), and the finish helper — which the
    # caller's code could reach by touching the sentinel in /tmp — is ignored,
    # so a task can never "succeed" without result.json and its promised metrics.
    ttl_exit_code = 124
    honor_finish_request = False

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        backend = select_backend_cls(config)
        return backend is not None and issubclass(backend, DockerSessionBackend)

    def run(self, task: ExecutorTask, out_dir: Path) -> PythonResult:  # type: ignore[override]
        spec = self.require_spec(task, PythonSpecStrict)
        cfg = self._python_config(spec)
        artifacts = out_dir / ARTIFACTS_DIR
        try:
            session = self._run_session(task, out_dir, cfg)
        except SessionExitError as exc:
            if exc.exit_code == self.ttl_exit_code:
                raise ExecutionError(
                    f"python task timed out after {cfg.ttl_sec:g}s"
                ) from exc
            if exc.exit_code == 137:
                raise ExecutionError(
                    "python task was killed (exit 137), most often by its "
                    "memory limit"
                ) from exc
            raise ExecutionError(_failure_message(artifacts, exc)) from exc
        return PythonResult(
            exit_code=session.exit_code,
            value=_read_json(artifacts / "result.json"),
            metrics=_read_json(artifacts / "metrics.json") or {},
        )

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
            "FLOWMESH_PY_OUTPUT": OUTPUT_PATH,
            "FLOWMESH_PY_REQUIREMENTS": json.dumps(spec.requirements or []),
            "FLOWMESH_PY_EMITS": json.dumps(spec.emits or []),
            "FLOWMESH_PY_UID": str(UNPRIVILEGED_UID),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
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
                    "mountPath": OUTPUT_PATH,
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
        cfg.network_disabled = spec.network == "none"
        cfg.hardened = True
        cfg.extra_files = {
            BOOTSTRAP_PATH: _BOOTSTRAP_SOURCE.read_bytes(),
            CODE_PATH: spec.code.encode(),
        }
        return cfg


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable %s from python task: %s", path.name, exc)
        return None


def _failure_message(artifacts: Path, exc: ExecutionError) -> str:
    """The caller's own error (from error.json) beats the bare exit code."""
    error = _read_json(artifacts / "error.json")
    if isinstance(error, dict) and error.get("message") is not None:
        return f"python task failed: {error.get('type')}: {error['message']}"
    return str(exc)
