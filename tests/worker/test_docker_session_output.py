"""DockerSession's in-container probes."""

from typing import Any
from unittest.mock import MagicMock

from shared.tasks.specs.ssh import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.ssh_session import SSHConfig
from worker.executors.ssh_session.backends.docker import DockerSession, SSHMountPlan
from worker.executors.ssh_session.config import SSHOutputConfig

OUTPUT = "/mnt/flowmesh/output"


def _cfg(max_bytes: int | None = None) -> SSHConfig:
    cfg = SSHConfig.from_spec(
        SSHSpecStrict.model_validate(
            {"taskType": "ssh", "interactive": False, "command": ["true"]}
        ),
        DEFAULT_WORKER_CONFIG,
    )
    cfg.output = SSHOutputConfig(mount_path=OUTPUT, max_bytes=max_bytes)
    return cfg


def _session(container: Any, max_bytes: int | None = None) -> DockerSession:
    plan = SSHMountPlan(
        volumes=[],
        staged_input_specs=[],
        create_dirs=[OUTPUT],
        direct_output_path=None,
        copy_output_path=OUTPUT,
        staged_inputs_dir=None,
        staged_inputs_volume=None,
    )
    return DockerSession(
        MagicMock(), container, "c", _cfg(max_bytes), plan, log_stream=None
    )


class TestProbes:
    def test_probes_run_without_a_shell_under_a_fixed_environment(self) -> None:
        container = MagicMock()
        container.exec_run.return_value = MagicMock(
            exit_code=0, output=b"1234\t/mnt/flowmesh/output\n"
        )
        session = _session(container)

        assert session._container_path_size(OUTPUT) == 1234
        session.finish_requested()
        session.established_connections()

        for call in container.exec_run.call_args_list:
            argv = call.args[0]
            assert argv[0] not in ("sh", "bash")
            env = call.kwargs["environment"]
            assert env["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin"
            assert env["LD_PRELOAD"] == ""
        assert container.exec_run.call_args_list[0].args[0] == [
            "du",
            "-sb",
            "--",
            OUTPUT,
        ]

    def test_size_survives_warnings_on_stderr(self) -> None:
        container = MagicMock()
        container.exec_run.return_value = MagicMock(
            exit_code=1, output=b"du: cannot read 'x': gone\n42\t/mnt/flowmesh/output\n"
        )
        assert _session(container)._container_path_size(OUTPUT) == 42
