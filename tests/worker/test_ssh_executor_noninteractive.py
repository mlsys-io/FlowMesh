"""Tests for non-interactive SSH executor behaviour."""

import io
import logging
import tarfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

import worker.executors.ssh_executor as ssh_executor_module
from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_executor import (
    _SSH_RUN_ENTRYPOINT_PATH,
    SSHConfig,
    SSHExecutor,
)


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-noninteractive",
        "workflow_id": "wf-1",
        "owner_id": "owner",
        "assigned_worker": "worker-1",
        "dispatched_at": "2026-03-22T00:00:00Z",
        "task": {
            "apiVersion": "mloc/v1",
            "kind": "Task",
            "metadata": {"name": "wf:batch"},
            "spec": {
                "taskType": "ssh",
                "interactive": False,
                "image": "python:3.12-slim",
                "command": ["python", "-c", "print(1)"],
                **spec_updates,
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


# ------------------------------------------------------------------ #
# SSHConfig tests
# ------------------------------------------------------------------ #


class TestSSHConfigFromSpec:
    def test_noninteractive_config(self) -> None:
        task = _task_message()
        cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
        assert cfg.interactive is False
        assert cfg.command == ["python", "-c", "print(1)"]
        assert cfg.image == "python:3.12-slim"

    def test_interactive_config_default(self) -> None:
        spec = SSHSpecStrict.model_validate(
            {"taskType": "ssh", "authorizedKeys": ["ssh-rsa AAAA..."]}
        )
        cfg = SSHConfig.from_spec(spec, DEFAULT_WORKER_CONFIG)
        assert cfg.interactive is True
        assert cfg.command is None
        assert cfg.entrypoint is None

    def test_entrypoint_only(self) -> None:
        spec = SSHSpecStrict.model_validate(
            {
                "taskType": "ssh",
                "interactive": False,
                "image": "myimg",
                "entrypoint": ["/run.sh"],
            }
        )
        cfg = SSHConfig.from_spec(spec, DEFAULT_WORKER_CONFIG)
        assert cfg.interactive is False
        assert cfg.entrypoint == ["/run.sh"]
        assert cfg.command is None

    def test_command_and_entrypoint(self) -> None:
        spec = SSHSpecStrict.model_validate(
            {
                "taskType": "ssh",
                "image": "myimg",
                "entrypoint": ["/bin/bash", "-c"],
                "command": ["echo hello"],
            }
        )
        cfg = SSHConfig.from_spec(spec, DEFAULT_WORKER_CONFIG)
        assert cfg.interactive is False
        assert cfg.entrypoint == ["/bin/bash", "-c"]
        assert cfg.command == ["echo hello"]


# ------------------------------------------------------------------ #
# _resolve_noninteractive_command tests
# ------------------------------------------------------------------ #


class TestResolveNoninteractiveCommand:
    def _make_executor(self, tmp_path: Path) -> SSHExecutor:
        cfg = make_live_worker_config(tmp_path)
        return SSHExecutor(cfg, lifecycle=None)

    def test_command_only(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        cfg = SSHConfig.from_spec(
            SSHSpecStrict.model_validate(
                {
                    "taskType": "ssh",
                    "image": "python:3.12",
                    "command": ["python", "train.py"],
                }
            ),
            DEFAULT_WORKER_CONFIG,
        )
        client = MagicMock()
        result = executor._resolve_noninteractive_command(client, cfg)
        assert result == ["python", "train.py"]

    def test_entrypoint_only(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        cfg = SSHConfig.from_spec(
            SSHSpecStrict.model_validate(
                {
                    "taskType": "ssh",
                    "image": "myimg",
                    "entrypoint": ["/run.sh"],
                }
            ),
            DEFAULT_WORKER_CONFIG,
        )
        client = MagicMock()
        result = executor._resolve_noninteractive_command(client, cfg)
        assert result == ["/run.sh"]

    def test_entrypoint_and_command(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        cfg = SSHConfig.from_spec(
            SSHSpecStrict.model_validate(
                {
                    "taskType": "ssh",
                    "image": "myimg",
                    "entrypoint": ["/bin/bash", "-c"],
                    "command": ["echo hello"],
                }
            ),
            DEFAULT_WORKER_CONFIG,
        )
        client = MagicMock()
        result = executor._resolve_noninteractive_command(client, cfg)
        assert result == ["/bin/bash", "-c", "echo hello"]

    def test_neither_inspects_image(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        cfg = SSHConfig.from_spec(
            SSHSpecStrict.model_validate(
                {
                    "taskType": "ssh",
                    "interactive": False,
                    "image": "myimg:latest",
                }
            ),
            DEFAULT_WORKER_CONFIG,
        )
        mock_image = MagicMock()
        mock_image.attrs = {
            "Config": {
                "Entrypoint": ["/usr/bin/myapp"],
                "Cmd": ["--serve"],
            }
        }
        client = MagicMock()
        client.images.get.return_value = mock_image
        result = executor._resolve_noninteractive_command(client, cfg)
        assert result == ["/usr/bin/myapp", "--serve"]
        client.images.get.assert_called_once_with("myimg:latest")

    def test_neither_set_no_image_entrypoint_raises(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        cfg = SSHConfig.from_spec(
            SSHSpecStrict.model_validate(
                {
                    "taskType": "ssh",
                    "interactive": False,
                    "image": "emptyimg",
                }
            ),
            DEFAULT_WORKER_CONFIG,
        )
        mock_image = MagicMock()
        mock_image.attrs = {"Config": {"Entrypoint": None, "Cmd": None}}
        client = MagicMock()
        client.images.get.return_value = mock_image
        with pytest.raises(Exception, match="no Entrypoint or Cmd"):
            executor._resolve_noninteractive_command(client, cfg)


# ------------------------------------------------------------------ #
# _build_environment tests
# ------------------------------------------------------------------ #


class TestBuildEnvironment:
    def _make_executor(self, tmp_path: Path) -> SSHExecutor:
        cfg = make_live_worker_config(tmp_path)
        return SSHExecutor(cfg, lifecycle=None)

    def test_interactive_includes_ssh_vars(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        env = executor._build_environment(
            "flowmesh",
            ["ssh-rsa AAAA..."],
            {},
            [],
            [],
            interactive=True,
        )
        assert "SSH_USER" in env
        assert "AUTHORIZED_KEYS" in env
        assert "SSH_UID" in env

    def test_noninteractive_omits_ssh_vars(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        env = executor._build_environment(
            "flowmesh",
            ["ssh-rsa AAAA..."],
            {"MY_VAR": "val"},
            [],
            [],
            interactive=False,
        )
        assert "SSH_USER" not in env
        assert "AUTHORIZED_KEYS" not in env
        assert "SSH_UID" not in env
        assert env["MY_VAR"] == "val"
        assert "FLOWMESH_FINISH_SENTINEL" in env


# ------------------------------------------------------------------ #
# _build_run_kwargs tests
# ------------------------------------------------------------------ #


def _build_ssh_config(
    image: str = "myimg:latest",
    *,
    cpu_limit: float | None = None,
    memory_limit_bytes: int | None = None,
    pids_limit: int | None = None,
) -> SSHConfig:
    """Construct a minimal SSHConfig for _build_run_kwargs tests."""
    return SSHConfig(
        image=image,
        interactive=False,
        user="flowmesh",
        authorized_keys=[],
        command=None,
        entrypoint=None,
        ttl_sec=60.0,
        idle_sec=30.0,
        access_mode="direct",
        extra_env={},
        inputs=[],
        output=None,
        mounts=[],
        poll_interval_sec=1.0,
        stop_timeout_sec=5.0,
        cpu_limit=cpu_limit,
        memory_limit_bytes=memory_limit_bytes,
        pids_limit=pids_limit,
    )


class TestBuildRunKwargs:
    def _make_executor(
        self, tmp_path: Path, docker_gpu_runtime: str | None = None
    ) -> SSHExecutor:
        cfg = make_live_worker_config(tmp_path, docker_gpu_runtime=docker_gpu_runtime)
        return SSHExecutor(cfg, lifecycle=None)

    def test_noninteractive_injects_wrapper_entrypoint_and_command(
        self, tmp_path: Path
    ) -> None:
        executor = self._make_executor(tmp_path)
        kwargs = executor._build_run_kwargs(
            _build_ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={},
            volumes=["/host:/container:rw"],
            command=["python", "train.py"],
            interactive=False,
        )
        assert kwargs["entrypoint"] == [_SSH_RUN_ENTRYPOINT_PATH]
        assert kwargs["command"] == ["python", "train.py"]

    def test_interactive_does_not_override_image_entrypoint(
        self, tmp_path: Path
    ) -> None:
        executor = self._make_executor(tmp_path)
        kwargs = executor._build_run_kwargs(
            _build_ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={"22/tcp": None},
            volumes=["/host:/container:rw"],
            command=None,
            interactive=True,
        )
        assert "entrypoint" not in kwargs
        assert "command" not in kwargs

    def test_gpu_run_kwargs_omit_runtime_when_not_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path)
        fake_device_request = MagicMock(name="device_request")
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0")
        monkeypatch.setattr(
            ssh_executor_module,
            "DeviceRequest",
            MagicMock(return_value=fake_device_request),
        )

        kwargs = executor._build_run_kwargs(
            _build_ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={},
            volumes=[],
            command=["python", "train.py"],
            interactive=False,
        )

        assert kwargs["device_requests"] == [fake_device_request]
        assert "runtime" not in kwargs

    def test_gpu_run_kwargs_include_configured_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path, docker_gpu_runtime="nvidia")
        fake_device_request = MagicMock(name="device_request")
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0")
        monkeypatch.setattr(
            ssh_executor_module,
            "DeviceRequest",
            MagicMock(return_value=fake_device_request),
        )

        kwargs = executor._build_run_kwargs(
            _build_ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={},
            volumes=[],
            command=["python", "train.py"],
            interactive=False,
        )

        assert kwargs["device_requests"] == [fake_device_request]
        assert kwargs["runtime"] == "nvidia"

    def test_resource_limits_absent_when_unset(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        kwargs = executor._build_run_kwargs(
            _build_ssh_config(),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={},
            volumes=[],
            command=None,
            interactive=False,
        )
        assert "nano_cpus" not in kwargs
        assert "mem_limit" not in kwargs
        assert "pids_limit" not in kwargs

    def test_resource_limits_applied(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        kwargs = executor._build_run_kwargs(
            _build_ssh_config(
                cpu_limit=2.5,
                memory_limit_bytes=8 * 1024**3,
                pids_limit=256,
            ),
            container_name="worker-1_ssh-task-1234",
            environment={},
            labels={},
            ports={},
            volumes=[],
            command=None,
            interactive=False,
        )
        assert kwargs["nano_cpus"] == int(2.5 * 1_000_000_000)
        assert kwargs["mem_limit"] == 8 * 1024**3
        assert kwargs["pids_limit"] == 256


class TestNoninteractiveContainerStartup:
    def _make_executor(self, tmp_path: Path) -> SSHExecutor:
        cfg = make_live_worker_config(tmp_path)
        return SSHExecutor(cfg, lifecycle=None)

    def test_build_ssh_run_archive_contains_executable_script(
        self, tmp_path: Path
    ) -> None:
        executor = self._make_executor(tmp_path)
        archive_bytes = executor._build_ssh_run_archive()
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as tar:
            member = tar.getmember(_SSH_RUN_ENTRYPOINT_PATH.lstrip("/"))
            extracted = tar.extractfile(member)
            assert extracted is not None
            script = extracted.read().decode()
        assert member.mode == 0o755
        assert "flowmesh-finish" in script

    def test_create_and_start_noninteractive_container_injects_before_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path)
        monkeypatch.setattr(ssh_executor_module, "Container", MagicMock)
        container = MagicMock()
        client = MagicMock()
        client.containers.create.return_value = container
        kwargs = {"image": "myimg:latest", "entrypoint": [_SSH_RUN_ENTRYPOINT_PATH]}

        result, log_stream = executor._run_noninteractive_container(client, kwargs)

        assert result is container
        assert log_stream is container.attach.return_value
        client.containers.create.assert_called_once_with(**kwargs)
        container.put_archive.assert_called_once()
        container.attach.assert_called_once_with(
            stream=True, logs=True, stdout=True, stderr=True, demux=True
        )
        container.start.assert_called_once_with()
        container.remove.assert_not_called()

    def test_create_and_start_noninteractive_container_cleans_up_on_injection_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path)
        monkeypatch.setattr(ssh_executor_module, "Container", MagicMock)
        container = MagicMock()
        container.put_archive.side_effect = RuntimeError("boom")
        client = MagicMock()
        client.containers.create.return_value = container

        with pytest.raises(Exception, match="initialize non-interactive container"):
            executor._run_noninteractive_container(client, {"image": "x"})

        container.start.assert_not_called()
        container.remove.assert_called_once_with(force=True)

    def test_pulls_missing_image_and_retries_create(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path)
        monkeypatch.setattr(ssh_executor_module, "Container", MagicMock)
        container = MagicMock()
        client = MagicMock()
        client.containers.create.side_effect = [
            RuntimeError(
                '404 Client Error: Not Found ("No such image: python:3.12-slim")'
            ),
            container,
        ]

        result, log_stream = executor._start_container(
            client,
            {"image": "python:3.12-slim", "entrypoint": [_SSH_RUN_ENTRYPOINT_PATH]},
            interactive=False,
        )

        assert result is container
        assert log_stream is container.attach.return_value
        client.images.pull.assert_called_once_with("python:3.12-slim")
        assert client.containers.create.call_count == 2
        container.put_archive.assert_called_once()
        container.start.assert_called_once_with()

    def test_pulls_missing_image_and_retries_interactive_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._make_executor(tmp_path)
        monkeypatch.setattr(ssh_executor_module, "Container", MagicMock)
        container = MagicMock()
        client = MagicMock()
        client.containers.run.side_effect = [
            RuntimeError('404 Client Error: Not Found ("No such image: myimg:latest")'),
            container,
        ]

        result, log_stream = executor._start_container(
            client,
            {"image": "myimg:latest", "command": ["sleep", "1"]},
            interactive=True,
        )

        assert result is container
        assert log_stream is None
        client.images.pull.assert_called_once_with("myimg:latest")
        assert client.containers.run.call_count == 2


# ------------------------------------------------------------------ #
# _stream_container_logs tests
# ------------------------------------------------------------------ #


class TestStreamContainerLogs:
    def test_streams_stdout_and_stderr_as_log_records(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log_stream = iter(
            [
                (b"hello from stdout\n", None),
                (None, b"oops on stderr\n"),
                (b"line2\n", None),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=ssh_executor_module.__name__):
            SSHExecutor._stream_container_logs(log_stream)
        messages = [
            r.message for r in caplog.records if r.name == ssh_executor_module.__name__
        ]
        assert "hello from stdout" in messages
        assert "oops on stderr" in messages
        assert "line2" in messages
        streams = {
            r.message: getattr(r, "flowmesh_stream", None)
            for r in caplog.records
            if r.name == ssh_executor_module.__name__
        }
        assert streams["hello from stdout"] == "stdout"
        assert streams["oops on stderr"] == "stderr"

    def test_stderr_logged_at_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log_stream = iter(
            [
                (b"info line\n", None),
                (None, b"warn line\n"),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=ssh_executor_module.__name__):
            SSHExecutor._stream_container_logs(log_stream)

        levels = {
            r.message: r.levelno
            for r in caplog.records
            if r.name == ssh_executor_module.__name__
        }
        assert levels["info line"] == logging.INFO
        assert levels["warn line"] == logging.WARNING

    def test_buffers_partial_lines(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chunks that don't end with \\n are buffered until the next chunk."""
        log_stream = iter(
            [
                (b"hello ", None),
                (b"world\nsecond", None),
                (b" line\n", None),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=ssh_executor_module.__name__):
            SSHExecutor._stream_container_logs(log_stream)

        messages = [
            r.message for r in caplog.records if r.name == ssh_executor_module.__name__
        ]
        assert messages == ["hello world", "second line"]

    def test_flushes_unterminated_remainder(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A trailing partial line with no final \\n is emitted on stream end."""
        log_stream = iter(
            [
                (b"complete\n", None),
                (b"no newline at end", None),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=ssh_executor_module.__name__):
            SSHExecutor._stream_container_logs(log_stream)

        messages = [
            r.message for r in caplog.records if r.name == ssh_executor_module.__name__
        ]
        assert messages == ["complete", "no newline at end"]

    def test_drains_after_container_exits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The thread processes all remaining chunks — no early exit."""
        log_stream = iter(
            [
                (b"line1\n", None),
                (b"line2\n", None),
                (b"line3\n", None),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=ssh_executor_module.__name__):
            SSHExecutor._stream_container_logs(log_stream)

        messages = [
            r.message for r in caplog.records if r.name == ssh_executor_module.__name__
        ]
        assert messages == ["line1", "line2", "line3"]

    def test_handles_generator_exception_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _exploding_attach(**_: object):  # type: ignore[no-untyped-def]
            yield (b"first\n", None)
            raise RuntimeError("docker socket gone")

        log_stream = _exploding_attach()

        with caplog.at_level(logging.DEBUG):
            SSHExecutor._stream_container_logs(log_stream)

        messages = [
            r.message for r in caplog.records if r.name == ssh_executor_module.__name__
        ]
        assert "first" in messages


# ------------------------------------------------------------------ #
# _wait_for_port tests
# ------------------------------------------------------------------ #


class TestWaitForPort:
    def _make_executor(self, tmp_path: Path) -> SSHExecutor:
        cfg = make_live_worker_config(tmp_path)
        return SSHExecutor(cfg, lifecycle=None)

    def test_reports_container_exit_with_logs(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        container = MagicMock()
        container.name = "test-container"
        container.status = "exited"
        container.reload.return_value = None
        container.wait.return_value = {"StatusCode": 127}
        container.logs.return_value = b"bash: sshd: command not found\n"

        with pytest.raises(ExecutionError, match="exited.*code 127") as exc_info:
            executor._wait_for_port(container, timeout_sec=1)
        assert "sshd: command not found" in str(exc_info.value)

    def test_reports_container_exit_without_logs(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        container = MagicMock()
        container.name = "test-container"
        container.status = "exited"
        container.reload.return_value = None
        container.wait.return_value = {"StatusCode": 1}
        container.logs.side_effect = RuntimeError("no logs")

        with pytest.raises(ExecutionError, match="exited.*code 1"):
            executor._wait_for_port(container, timeout_sec=1)

    def test_timeout_message_suggests_sshd(self, tmp_path: Path) -> None:
        executor = self._make_executor(tmp_path)
        container = MagicMock()
        container.name = "test-container"
        container.status = "running"
        container.reload.return_value = None
        container.ports = {}

        with pytest.raises(ExecutionError, match="openssh-server") as exc_info:
            executor._wait_for_port(container, timeout_sec=0.1)
        assert "omitting the image field" in str(exc_info.value)
