"""DockerSession's in-container probes and the output it copies out."""

import io
import tarfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.tasks.specs.ssh import SSHSpecStrict
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.base_executor import ExecutionError
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


def _archive(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo("output")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(f"output/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"output/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return stream.getvalue()


def _chunked(data: bytes, size: int = 700) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


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


class TestCollectOutput:
    def _collect(
        self, tmp_path: Path, archive: bytes, max_bytes: int | None = None
    ) -> Path:
        container = MagicMock()
        container.get_archive.return_value = (iter(_chunked(archive)), {})
        destination = tmp_path / "artifacts"
        _session(container, max_bytes).collect_output(destination)
        return destination

    def test_regular_files_are_copied_and_links_dropped(self, tmp_path: Path) -> None:
        out = self._collect(
            tmp_path,
            _archive({"a.txt": b"hello", "sub/b.txt": b"x" * 3000}, {"leak": "/etc"}),
        )
        assert (out / "a.txt").read_bytes() == b"hello"
        assert (out / "sub/b.txt").read_bytes() == b"x" * 3000
        assert not (out / "leak").exists() and not (out / "leak").is_symlink()

    def test_output_past_max_bytes_is_refused(self, tmp_path: Path) -> None:
        archive = _archive({"a.bin": b"a" * 600, "b.bin": b"b" * 600})
        with pytest.raises(ExecutionError, match=r"exceeded maxBytes \(1200 > 1000\)"):
            self._collect(tmp_path, archive, max_bytes=1000)
        assert not (tmp_path / "artifacts/b.bin").exists()

    def test_output_within_max_bytes_is_kept(self, tmp_path: Path) -> None:
        out = self._collect(tmp_path, _archive({"a.bin": b"a" * 600}), max_bytes=600)
        assert (out / "a.bin").stat().st_size == 600


class TestDiskUsage:
    def test_reads_the_container_layer_size(self) -> None:
        container = MagicMock(id="cid")
        session = _session(container)
        api = cast(MagicMock, session._client).api
        api.containers.return_value = [{"SizeRw": 4096, "SizeRootFs": 1 << 30}]

        assert session.disk_usage_bytes() == 4096
        api.containers.assert_called_once_with(
            all=True, size=True, filters={"id": "cid"}
        )

    def test_unreadable_size_is_unknown_not_zero(self) -> None:
        session = _session(MagicMock(id="cid"))
        api = cast(MagicMock, session._client).api
        api.containers.return_value = []
        assert session.disk_usage_bytes() is None
        api.containers.return_value = [{"Id": "cid"}]
        assert session.disk_usage_bytes() is None
        api.containers.side_effect = RuntimeError("daemon gone")
        assert session.disk_usage_bytes() is None
