"""Tests for ``worker.executors.utils.distributed.run_torchrun``."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from worker.executors.utils import distributed


@pytest.fixture
def captured_env():
    """Capture os.environ inside torchrun_main and surface it for assertions."""

    captured: dict[str, str | None] = {}

    def _fake_main(argv: list[str]) -> None:
        captured["argv"] = argv  # type: ignore[assignment]
        captured["PYTHONPATH"] = os.environ.get("PYTHONPATH")
        captured["FLAG"] = os.environ.get("KV_TEST_LAUNCHER")

    with patch.object(distributed, "_torchrun_main", side_effect=_fake_main):
        yield captured


def test_run_torchrun_passes_argv(captured_env: dict[str, str | None]) -> None:
    distributed.run_torchrun(
        nproc_per_node=4,
        module="worker.executors.sft_dist_entry",
        module_args=["/tmp/task.json", "/tmp/out"],
        launcher_env_flag="KV_TEST_LAUNCHER",
    )

    assert captured_env["argv"] == [
        "--nproc_per_node",
        "4",
        "-m",
        "worker.executors.sft_dist_entry",
        "/tmp/task.json",
        "/tmp/out",
    ]


def test_run_torchrun_sets_launcher_flag(captured_env: dict[str, str | None]) -> None:
    distributed.run_torchrun(
        nproc_per_node=2,
        module="worker.executors.dpo_dist_entry",
        module_args=["a", "b"],
        launcher_env_flag="KV_TEST_LAUNCHER",
    )

    assert captured_env["FLAG"] == "1"


def test_run_torchrun_prepends_repo_root_to_pythonpath(
    captured_env: dict[str, str | None],
) -> None:
    src_root = Path(distributed.__file__).resolve().parents[3].as_posix()

    with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
        distributed.run_torchrun(
            nproc_per_node=1,
            module="worker.executors.sft_dist_entry",
            module_args=[],
            launcher_env_flag="KV_TEST_LAUNCHER",
        )

    pythonpath = captured_env["PYTHONPATH"]
    assert pythonpath is not None
    parts = pythonpath.split(os.pathsep)
    assert parts[0] == src_root
    assert "/existing/path" in parts


def test_run_torchrun_pythonpath_when_unset(
    captured_env: dict[str, str | None],
) -> None:
    src_root = Path(distributed.__file__).resolve().parents[3].as_posix()

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PYTHONPATH", None)
        distributed.run_torchrun(
            nproc_per_node=1,
            module="worker.executors.sft_dist_entry",
            module_args=[],
            launcher_env_flag="KV_TEST_LAUNCHER",
        )

    assert captured_env["PYTHONPATH"] == src_root


def test_run_torchrun_restores_env_on_success(
    captured_env: dict[str, str | None],
) -> None:
    with patch.dict(os.environ, {"PYTHONPATH": "/before"}, clear=False):
        os.environ.pop("KV_TEST_LAUNCHER", None)

        distributed.run_torchrun(
            nproc_per_node=1,
            module="worker.executors.sft_dist_entry",
            module_args=[],
            launcher_env_flag="KV_TEST_LAUNCHER",
        )

        assert os.environ["PYTHONPATH"] == "/before"
        assert "KV_TEST_LAUNCHER" not in os.environ


def test_run_torchrun_restores_env_on_exception() -> None:
    def _boom(_: list[str]) -> None:
        raise RuntimeError("torchrun crashed")

    with patch.object(distributed, "_torchrun_main", side_effect=_boom):
        with patch.dict(os.environ, {"PYTHONPATH": "/before"}, clear=False):
            os.environ.pop("KV_TEST_LAUNCHER", None)

            with pytest.raises(RuntimeError, match="torchrun crashed"):
                distributed.run_torchrun(
                    nproc_per_node=1,
                    module="worker.executors.sft_dist_entry",
                    module_args=[],
                    launcher_env_flag="KV_TEST_LAUNCHER",
                )

            assert os.environ["PYTHONPATH"] == "/before"
            assert "KV_TEST_LAUNCHER" not in os.environ
