"""Tests for ``worker.executors.utils.distributed`` launchers."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from shared.tasks import TaskType
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_worker_task_message
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

    # ``--tee 3`` is required so rank stdout/stderr surface on the parent
    # console and land in the elastic per-rank log dir; without it a rank
    # crash is reported only as an opaque ChildFailedError.
    assert captured_env["argv"] == [
        "--nproc_per_node",
        "4",
        "--tee",
        "3",
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


@pytest.fixture
def fake_deepspeed():
    """Inject a stub ``deepspeed.launcher.runner`` so the GPU-only package is
    not required for these tests, and capture the argv passed to its main()."""

    captured: dict[str, object] = {}

    def _fake_main(argv: list[str]) -> None:
        captured["argv"] = argv
        captured["PYTHONPATH"] = os.environ.get("PYTHONPATH")
        captured["FLAG"] = os.environ.get("KV_TEST_LAUNCHER")

    runner = ModuleType("deepspeed.launcher.runner")
    runner.main = _fake_main  # type: ignore[attr-defined]
    launcher = ModuleType("deepspeed.launcher")
    launcher.runner = runner  # type: ignore[attr-defined]
    deepspeed_pkg = ModuleType("deepspeed")
    deepspeed_pkg.launcher = launcher  # type: ignore[attr-defined]

    saved = {
        "deepspeed": sys.modules.get("deepspeed"),
        "deepspeed.launcher": sys.modules.get("deepspeed.launcher"),
        "deepspeed.launcher.runner": sys.modules.get("deepspeed.launcher.runner"),
    }
    sys.modules["deepspeed"] = deepspeed_pkg
    sys.modules["deepspeed.launcher"] = launcher
    sys.modules["deepspeed.launcher.runner"] = runner
    try:
        yield captured
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_run_deepspeed_passes_argv(fake_deepspeed: dict[str, object]) -> None:
    distributed.run_deepspeed(
        num_gpus=4,
        module="worker.executors.sft_dist_entry",
        module_args=["/tmp/task.json", "/tmp/out"],
        launcher_env_flag="KV_TEST_LAUNCHER",
    )

    assert fake_deepspeed["argv"] == [
        "--num_gpus",
        "4",
        "--module",
        "worker.executors.sft_dist_entry",
        "/tmp/task.json",
        "/tmp/out",
    ]


def test_run_deepspeed_scopes_env(fake_deepspeed: dict[str, object]) -> None:
    src_root = Path(distributed.__file__).resolve().parents[3].as_posix()

    with patch.dict(os.environ, {"PYTHONPATH": "/before"}, clear=False):
        os.environ.pop("KV_TEST_LAUNCHER", None)

        distributed.run_deepspeed(
            num_gpus=2,
            module="worker.executors.sft_dist_entry",
            module_args=[],
            launcher_env_flag="KV_TEST_LAUNCHER",
        )

        assert fake_deepspeed["FLAG"] == "1"
        pythonpath = fake_deepspeed["PYTHONPATH"]
        assert isinstance(pythonpath, str)
        parts = pythonpath.split(os.pathsep)
        assert parts[0] == src_root
        assert "/before" in parts
        assert os.environ["PYTHONPATH"] == "/before"
        assert "KV_TEST_LAUNCHER" not in os.environ


def test_deepspeed_available_when_spec_resolves() -> None:
    fake_spec = object()
    with patch("importlib.util.find_spec", return_value=fake_spec):
        assert distributed.deepspeed_available() is True


def test_deepspeed_available_when_spec_missing() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        assert distributed.deepspeed_available() is False


def test_deepspeed_available_when_find_spec_raises() -> None:
    """DeepSpeed's package init probes CUDA ops eagerly; on a CUDA-less host
    `find_spec` raises before the spec is ever returned. Treat that as
    'not available' rather than letting it propagate."""

    def _boom(_name: str) -> object:
        raise RuntimeError("CUDA_HOME does not exist")

    with patch("importlib.util.find_spec", side_effect=_boom):
        assert distributed.deepspeed_available() is False


def test_task_spec_dump_load_round_trip(tmp_path: Path) -> None:
    """The on-disk task_spec.json must survive the executor → dist_entry hop.

    The training executors persist their ``WorkerTaskMessage`` via
    ``json.dump(task.model_dump(mode="json", by_alias=True), fh)``; the
    matching ``*_dist_entry`` rehydrates it via
    ``WorkerTaskMessage.model_validate(json.load(fh))``. ``WorkerTaskMessage``
    has a ``model_serializer`` that dedups string values in ``mode="json"``
    and a ``model_validator(mode="before")`` that restores the deduped form,
    so the round-trip works only when both sides go through the model.
    """
    # Spec doesn't matter for the contract — just needs to be a valid pydantic
    # spec; reuse echo to avoid pulling in TRL/torch in the import path.
    original = make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO, data={"items": ["a", "b"]}),
        api_version="flowmesh/v1",
        kind="EchoTask",
        task_id="tsk-roundtrip",
        workflow_id="wfl-roundtrip",
        owner_id="usr-roundtrip",
    )

    task_file = tmp_path / "task_spec.json"
    with task_file.open("w", encoding="utf-8") as fh:
        json.dump(original.model_dump(mode="json", by_alias=True), fh)

    with task_file.open("r", encoding="utf-8") as fh:
        rehydrated = WorkerTaskMessage.model_validate(json.load(fh))

    assert rehydrated.task_id == "tsk-roundtrip"
    assert rehydrated.workflow_id == "wfl-roundtrip"
    assert rehydrated.owner_id == "usr-roundtrip"
    # ``task.spec`` must be reachable — this is what each *_dist_entry feeds
    # into ``executor.run`` via ``self.require_spec(task, ...)``.
    assert rehydrated.spec is not None
    assert rehydrated.task.kind == "EchoTask"
