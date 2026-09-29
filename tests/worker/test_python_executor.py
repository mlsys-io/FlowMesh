"""The python executor: config it hands the session backend, and how it ends.

The bootstrap tests at the bottom run src/worker/docker/python-run.py for real
in a subprocess — the same file the container runs — as a non-root user, so the
privilege drop is skipped and everything else is exercised.
"""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from shared.schemas.result import SSHResult
from shared.tasks.specs import PythonSpecStrict, SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.executors.base_executor import ExecutionError
from worker.executors.python_executor import (
    BOOTSTRAP_PATH,
    CODE_PATH,
    OUTPUT_PATH,
    PythonExecutor,
)
from worker.executors.session_executor import (
    SessionEnd,
    SessionEndReason,
    SessionOutcome,
)
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import SSHConfig, SSHSession
from worker.executors.ssh_session.backends import docker as docker_backend_module
from worker.executors.ssh_session.backends.docker import DockerSessionBackend

BOOTSTRAP = Path(__file__).resolve().parents[2] / "src/worker/docker/python-run.py"


@pytest.fixture(autouse=True)
def _docker_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_backend_module, "docker_available", lambda: True)


def _spec(**updates: object) -> PythonSpecStrict:
    return cast(
        PythonSpecStrict,
        PythonSpecStrict.model_validate(
            {"taskType": "python", "code": "def main():\n    return 1\n", **updates}
        ),
    )


def _executor(tmp_path: Path) -> PythonExecutor:
    return PythonExecutor(make_live_worker_config(tmp_path))


# ------------------------------------------------------------------ #
# Config handed to the session backend
# ------------------------------------------------------------------ #


class TestPythonConfig:
    def test_hardened_offline_noninteractive(self, tmp_path: Path) -> None:
        cfg = _executor(tmp_path)._python_config(_spec())
        assert cfg.interactive is False
        assert cfg.network_disabled is True
        assert cfg.hardened is True
        assert cfg.command == ["python3", BOOTSTRAP_PATH]
        assert cfg.output is not None and cfg.output.mount_path == OUTPUT_PATH

    def test_code_and_bootstrap_travel_as_files_not_env(self, tmp_path: Path) -> None:
        code = "def main():\n    return 'x' * 3\n"
        cfg = _executor(tmp_path)._python_config(_spec(code=code))
        assert cfg.extra_files[CODE_PATH] == code.encode()
        assert cfg.extra_files[BOOTSTRAP_PATH] == BOOTSTRAP.read_bytes()
        assert all(code not in str(v) for v in cfg.extra_env.values())

    def test_bridge_network_keeps_network(self, tmp_path: Path) -> None:
        spec = _spec(network="bridge", requirements=["numpy"])
        cfg = _executor(tmp_path)._python_config(spec)
        assert cfg.network_disabled is False
        assert json.loads(str(cfg.extra_env["FLOWMESH_PY_REQUIREMENTS"])) == ["numpy"]

    def test_timeout_becomes_ttl(self, tmp_path: Path) -> None:
        cfg = _executor(tmp_path)._python_config(_spec(timeoutSeconds=42))
        assert cfg.ttl_sec == 42

    def test_inputs_map_stage_to_mount(self, tmp_path: Path) -> None:
        spec = _spec(
            inputs=[
                {"stage": "prep"},
                {"stage": "raw", "mountPath": "/mnt/flowmesh/r"},
            ],
            dependsOn=["prep", "raw"],
        )
        cfg = _executor(tmp_path)._python_config(spec)
        assert json.loads(str(cfg.extra_env["FLOWMESH_PY_INPUTS"])) == {
            "prep": "/mnt/flowmesh/inputs/prep",
            "raw": "/mnt/flowmesh/r",
        }
        assert [i.stage for i in cfg.inputs] == ["prep", "raw"]

    def test_user_env_and_emits_pass_through(self, tmp_path: Path) -> None:
        cfg = _executor(tmp_path)._python_config(
            _spec(env={"SEED": "7"}, emits=["score"])
        )
        assert cfg.extra_env["SEED"] == "7"
        assert json.loads(str(cfg.extra_env["FLOWMESH_PY_EMITS"])) == ["score"]


class TestGPUs:
    def test_no_gpu_block_means_no_gpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        executor = PythonExecutor(
            make_live_worker_config(tmp_path, enable_ssh_gpu_limit=False)
        )
        cfg = executor._python_config(_spec())
        assert cfg.gpu_device_ids == []
        assert cfg.extra_env["NVIDIA_VISIBLE_DEVICES"] == "void"

    def test_user_env_cannot_unhide_gpus(self, tmp_path: Path) -> None:
        cfg = _executor(tmp_path)._python_config(
            _spec(env={"NVIDIA_VISIBLE_DEVICES": "all"})
        )
        assert cfg.extra_env["NVIDIA_VISIBLE_DEVICES"] == "void"

    def test_declared_gpus_are_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKER_HOST_GPU_ID", "0,1")
        executor = PythonExecutor(
            make_live_worker_config(tmp_path, enable_ssh_gpu_limit=False)
        )
        cfg = executor._python_config(
            _spec(resources={"hardware": {"gpu": {"count": 1}}})
        )
        assert cfg.gpu_device_ids == ["0", "1"]
        assert "NVIDIA_VISIBLE_DEVICES" not in cfg.extra_env


class TestAvailability:
    def test_available_with_docker(self) -> None:
        assert PythonExecutor.is_available(DEFAULT_WORKER_CONFIG)

    def test_never_falls_back_to_the_process_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import worker.executors.python_executor as mod
        from worker.executors.ssh_session import ProcessSessionBackend

        monkeypatch.setattr(
            mod, "select_backend_cls", lambda cfg: ProcessSessionBackend
        )
        assert not PythonExecutor.is_available(DEFAULT_WORKER_CONFIG)

    def test_unavailable_without_any_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import worker.executors.python_executor as mod

        monkeypatch.setattr(mod, "select_backend_cls", lambda cfg: None)
        assert not PythonExecutor.is_available(DEFAULT_WORKER_CONFIG)


# ------------------------------------------------------------------ #
# Docker backend honours the hardening fields
# ------------------------------------------------------------------ #


def _base_cfg() -> SSHConfig:
    return SSHConfig.from_spec(
        SSHSpecStrict.model_validate(
            {"taskType": "ssh", "interactive": False, "command": ["true"]}
        ),
        DEFAULT_WORKER_CONFIG,
    )


class TestDockerHardening:
    def _kwargs(self, tmp_path: Path, cfg: SSHConfig) -> dict[str, object]:
        backend = DockerSessionBackend(make_live_worker_config(tmp_path))
        backend._ssh_network = "flowmesh_ssh"
        return backend._build_run_kwargs(cfg, "c", {}, {}, {}, [], ["true"], False)

    def test_ssh_defaults_unchanged(self, tmp_path: Path) -> None:
        kwargs = self._kwargs(tmp_path, _base_cfg())
        assert kwargs["network"] == "flowmesh_ssh"
        assert "network_mode" not in kwargs and "cap_drop" not in kwargs

    def test_network_disabled_and_hardened(self, tmp_path: Path) -> None:
        cfg = _base_cfg()
        cfg.network_disabled = True
        cfg.hardened = True
        kwargs = self._kwargs(tmp_path, cfg)
        assert kwargs["network_mode"] == "none"
        assert "network" not in kwargs
        assert kwargs["cap_drop"] == ["ALL"]
        assert "NET_RAW" not in cast(list[str], kwargs["cap_add"])

    def test_archive_carries_extra_files(self) -> None:
        archive = DockerSessionBackend._build_ssh_run_archive(
            {"/opt/flowmesh/task.py": b"print(1)\n"}
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            member = tar.getmember("opt/flowmesh/task.py")
            assert member.mode == 0o644
            assert tar.extractfile(member).read() == b"print(1)\n"  # type: ignore[union-attr]


# ------------------------------------------------------------------ #
# How a python session ends
# ------------------------------------------------------------------ #


class _Session(SSHSession):
    def __init__(self, exit_code: int | None = None, finish: bool = False) -> None:
        self.exit_code, self.finish, self.stopped = exit_code, finish, False

    def login_user(self) -> str:
        return "nobody"

    def wait_ready(self, timeout_sec: float) -> int:
        return 0

    def poll(self) -> int | None:
        return self.exit_code

    def finish_requested(self) -> bool:
        return self.finish

    def established_connections(self) -> int | None:
        return None

    def output_size_bytes(self) -> int | None:
        return None

    def collect_output(self, destination: Path) -> None:
        return None

    def stop(self, timeout_sec: float) -> None:
        self.stopped = True

    def cleanup(self) -> None:
        return None


def _outcome(reason: SessionEndReason, exit_code: int = 0) -> SessionOutcome:
    return SessionOutcome("ssn-1", SessionEnd(reason, exit_code))


class TestEnding:
    def _cfg(self, tmp_path: Path) -> SSHConfig:
        cfg = _executor(tmp_path)._python_config(_spec(timeoutSeconds=1))
        cfg.ttl_sec = 0.05
        cfg.poll_interval_sec = 0.01
        return cfg

    def test_ttl_stops_the_session(self, tmp_path: Path) -> None:
        session = _Session()
        end = _executor(tmp_path)._wait_for_session(session, self._cfg(tmp_path))
        assert end == SessionEnd("ttl")
        assert session.stopped

    def test_finish_helper_is_ignored(self, tmp_path: Path) -> None:
        session = _Session(finish=True)
        end = _executor(tmp_path)._wait_for_session(session, self._cfg(tmp_path))
        assert end == SessionEnd("ttl")  # ran to the deadline instead of "finishing"

    def test_finish_helper_still_ends_an_ssh_session(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        cfg.honor_finish_request = True
        executor = SSHExecutor(make_live_worker_config(tmp_path))
        assert executor._wait_for_session(_Session(finish=True), cfg) == SessionEnd(
            "finished"
        )

    def _run(
        self, tmp_path: Path, outcome: SessionOutcome, files: dict[str, str]
    ) -> str:
        executor = _executor(tmp_path)
        executor._run_session = MagicMock(return_value=outcome)  # type: ignore[method-assign]
        out = tmp_path / "out"
        (out / "artifacts").mkdir(parents=True)
        for name, text in files.items():
            (out / "artifacts" / name).write_text(text)
        executor.require_spec = MagicMock(return_value=_spec(timeoutSeconds=5))  # type: ignore[method-assign]
        with pytest.raises(ExecutionError) as info:
            executor.run(MagicMock(upstream_task_ids=None), out)
        return str(info.value)

    def test_timeout_message(self, tmp_path: Path) -> None:
        msg = self._run(tmp_path, _outcome("ttl"), {})
        assert msg == "python task timed out after 5s"

    def test_exit_124_is_not_a_timeout(self, tmp_path: Path) -> None:
        msg = self._run(tmp_path, _outcome("exited", 124), {})
        assert msg == "python task exited with code 124"

    def test_callers_error_beats_exit_code(self, tmp_path: Path) -> None:
        error = json.dumps({"type": "ValueError", "message": "bad input"})
        msg = self._run(tmp_path, _outcome("exited", 1), {"error.json": error})
        assert msg == "python task failed: ValueError: bad input"

    @pytest.mark.parametrize("reason", ["finished", "lost", "idle"])
    def test_any_other_ending_is_a_failure(
        self, tmp_path: Path, reason: SessionEndReason
    ) -> None:
        # Even with a result on disk: only the process exiting 0 is success.
        msg = self._run(tmp_path, _outcome(reason), {"result.json": "1"})
        assert msg.startswith("python task")

    def test_success_reads_result_and_metrics(self, tmp_path: Path) -> None:
        executor = _executor(tmp_path)
        out = tmp_path / "out"
        (out / "artifacts").mkdir(parents=True)
        (out / "artifacts/result.json").write_text('{"answer": 42}')
        (out / "artifacts/metrics.json").write_text('{"score": 0.5}')
        executor._run_session = MagicMock(return_value=_outcome("exited", 0))  # type: ignore[method-assign]
        executor.require_spec = MagicMock(return_value=_spec())  # type: ignore[method-assign]
        result = executor.run(MagicMock(upstream_task_ids=None), out)
        assert result.value == {"answer": 42}
        assert result.metrics == {"score": 0.5}
        assert result.task_type == "python"


class TestSSHEnding:
    def _run(self, tmp_path: Path, outcome: SessionOutcome) -> SSHResult:
        executor = SSHExecutor(make_live_worker_config(tmp_path))
        executor._run_session = MagicMock(return_value=outcome)  # type: ignore[method-assign]
        executor.require_spec = MagicMock(  # type: ignore[method-assign]
            return_value=SSHSpecStrict.model_validate(
                {"taskType": "ssh", "interactive": False, "command": ["true"]}
            )
        )
        return executor.run(MagicMock(), tmp_path / "out")

    @pytest.mark.parametrize("reason", ["ttl", "finished", "lost"])
    def test_non_exit_endings_are_success(
        self, tmp_path: Path, reason: SessionEndReason
    ) -> None:
        assert self._run(tmp_path, _outcome(reason)).exit_code == 0

    def test_nonzero_exit_fails(self, tmp_path: Path) -> None:
        with pytest.raises(
            ExecutionError, match="Non-interactive session exited with code 3"
        ):
            self._run(tmp_path, _outcome("exited", 3))


# ------------------------------------------------------------------ #
# The bootstrap, run for real
# ------------------------------------------------------------------ #


def _bootstrap(tmp_path: Path, code: str, **env: str) -> tuple[int, Path]:
    out = tmp_path / "output"
    code_file = tmp_path / "task.py"
    code_file.write_text(code)
    proc = subprocess.run(
        [sys.executable, str(BOOTSTRAP)],
        env={
            "PATH": os.environ.get("PATH", ""),
            "FLOWMESH_PY_CODE": str(code_file),
            "FLOWMESH_PY_OUTPUT": str(out),
            "FLOWMESH_PY_UID": str(os.getuid()),
            **env,
        },
        capture_output=True,
        timeout=60,
    )
    return proc.returncode, out


def _load(out: Path, name: str) -> object:
    return json.loads((out / name).read_text())


@pytest.mark.skipif(os.getuid() == 0, reason="bootstrap would drop privileges")
class TestBootstrap:
    def test_return_value_and_metrics(self, tmp_path: Path) -> None:
        code = (
            "def main(inputs):\n"
            "    return {'n': len(inputs), 'metrics': {'score': 3}}\n"
        )
        rc, out = _bootstrap(
            tmp_path,
            code,
            FLOWMESH_PY_INPUTS='{"a": "/x"}',
            FLOWMESH_PY_EMITS='["score"]',
        )
        assert rc == 0
        assert _load(out, "result.json") == {"n": 1, "metrics": {"score": 3}}
        assert _load(out, "metrics.json") == {"score": 3.0}

    def test_no_arg_entrypoint_and_written_metrics_merge(self, tmp_path: Path) -> None:
        code = (
            "import json, os\n"
            "def go():\n"
            "    p = os.path.join(os.environ['FLOWMESH_OUTPUT'], 'metrics.json')\n"
            "    json.dump({'a': 1}, open(p, 'w'))\n"
            "    return {'metrics': {'b': 2}}\n"
        )
        rc, out = _bootstrap(tmp_path, code, FLOWMESH_PY_ENTRYPOINT="go")
        assert rc == 0
        assert _load(out, "metrics.json") == {"a": 1.0, "b": 2.0}

    def test_exception_writes_error_and_fails(self, tmp_path: Path) -> None:
        rc, out = _bootstrap(tmp_path, "def main():\n    raise ValueError('boom')\n")
        assert rc == 1
        error = cast(dict[str, str], _load(out, "error.json"))
        assert (error["type"], error["message"]) == ("ValueError", "boom")
        assert "Traceback" in error["traceback"]

    def test_missing_promised_metric_fails(self, tmp_path: Path) -> None:
        rc, out = _bootstrap(
            tmp_path, "def main():\n    return {}\n", FLOWMESH_PY_EMITS='["score"]'
        )
        assert rc == 3
        assert "score" in cast(dict[str, str], _load(out, "error.json"))["message"]

    def test_unserialisable_result_fails(self, tmp_path: Path) -> None:
        rc, out = _bootstrap(tmp_path, "def main():\n    return object()\n")
        assert rc == 3
        assert cast(dict[str, str], _load(out, "error.json"))["type"] == "ResultError"

    def test_non_numeric_metric_fails(self, tmp_path: Path) -> None:
        rc, _ = _bootstrap(
            tmp_path, "def main():\n    return {'metrics': {'s': 'hi'}}\n"
        )
        assert rc == 3

    def test_missing_entrypoint_fails(self, tmp_path: Path) -> None:
        rc, out = _bootstrap(tmp_path, "x = 1\n")
        assert rc == 3
        assert (
            cast(dict[str, str], _load(out, "error.json"))["type"] == "EntrypointError"
        )


def test_inputs_default_to_the_resolved_upstream_stages(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    seen: dict[str, object] = {}

    def _capture(task: object, out_dir: Path, cfg: SSHConfig) -> SessionOutcome:
        seen["inputs"] = [i.stage for i in cfg.inputs]
        seen["env"] = json.loads(str(cfg.extra_env["FLOWMESH_PY_INPUTS"]))
        return _outcome("exited", 0)

    executor._run_session = _capture  # type: ignore[method-assign]
    executor.require_spec = MagicMock(return_value=_spec())  # type: ignore[method-assign]
    task = MagicMock(upstream_task_ids={"prep": "t-a", "raw": "t-b"})
    (tmp_path / "out" / "artifacts").mkdir(parents=True)
    executor.run(task, tmp_path / "out")
    assert seen["inputs"] == ["prep", "raw"]
    assert seen["env"] == {
        "prep": "/mnt/flowmesh/inputs/prep",
        "raw": "/mnt/flowmesh/inputs/raw",
    }
