"""SSH session executor.

Supports two modes:

**Interactive** (default): Creates an ephemeral session running sshd, emits a
TASK_UPDATE event with connection info, and blocks until the session ends (TTL,
idle timeout, or worker shutdown).

**Non-interactive** (``interactive=false``): Runs a user-provided container
image with an optional custom entrypoint/command.

A session backend (``worker.executors.ssh_session``) supplies the sandbox the
session runs in, selected by ``SSH_SESSION_BACKEND``.
"""

from pathlib import Path

from shared.schemas.result import SSHResult
from shared.tasks.specs.ssh import SSHSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors.ssh_session import (
    ResolvedSSHInput,
    SSHConfig,
    SSHOutputConfig,
)

from .base_executor import ExecutionError, ExecutorTask
from .session_executor import SessionExecutor

__all__ = ["ResolvedSSHInput", "SSHConfig", "SSHExecutor", "SSHOutputConfig"]


class SSHExecutor(SessionExecutor):
    """Executor for SSH tasks (interactive sessions and non-interactive jobs)."""

    name = "ssh"
    supported_task_types = frozenset({TaskType.SSH})

    def run(self, task: ExecutorTask, out_dir: Path) -> SSHResult:
        spec = self.require_spec(task, SSHSpecStrict)
        cfg = self._config_for(spec)
        outcome = self._run_session(task, out_dir, cfg)
        exit_code = outcome.end.exit_code

        result = SSHResult(session_id=outcome.session_id, exit_code=exit_code)
        if cfg.interactive:
            for key, value in outcome.ready_info.items():
                setattr(result, key, value)
            return result

        if cfg.command is not None:
            result.command = cfg.command
        if cfg.entrypoint is not None:
            result.entrypoint = cfg.entrypoint
        if exit_code != 0:
            raise ExecutionError(
                f"Non-interactive session exited with code {exit_code}"
            )
        return result
