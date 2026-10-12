"""DiffusersExecutor renders merged children in the parent's pipeline call.

FlowMesh merges ``diffusion`` tasks that differ only in their prompts, so the
executor must run every merged child too: one pipeline call over all prompts,
each task's images in its own output dir, and each child's result in
``result.children``. Driven through the public ``run`` with a fake pipeline.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from PIL import Image  # noqa: E402

from shared.schemas.result import DiffusionResult  # noqa: E402
from shared.tasks import MergedChildTaskStrict  # noqa: E402
from shared.tasks.components.model import ModelConfig, ModelSource  # noqa: E402
from shared.tasks.specs import DiffusionSpecStrict, EchoSpecStrict  # noqa: E402
from shared.tasks.task_type import TaskType  # noqa: E402
from tests.worker.factories import (  # noqa: E402
    DEFAULT_WORKER_CONFIG,
    make_worker_task_message,
)
from worker.executors import diffusers_executor as mod  # noqa: E402
from worker.executors.base_executor import ExecutionError, RunControl  # noqa: E402
from worker.executors.diffusers_executor import DiffusersExecutor  # noqa: E402


class _FakePipe:
    """Returns one solid image per requested prompt, shaded by its position."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        count = len(kwargs["prompt_embeds"])
        self.batch_sizes.append(count)
        return SimpleNamespace(
            images=[Image.new("L", (4, 4), color=i) for i in range(count)]
        )


def _spec(items: list[str]) -> DiffusionSpecStrict:
    return DiffusionSpecStrict(
        taskType=TaskType.DIFFUSION,
        model=ModelConfig(source=ModelSource(identifier="org/sd"), diffusers={}),
        data={"type": "list", "items": items},
        inference={"num_inference_steps": 2},
    )


def _child(task_id: str, items: list[str]) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id="usr-other",
        workflow_id="wfl-other",
        spec=_spec(items),
    )


def _executor(monkeypatch: pytest.MonkeyPatch, pipe: _FakePipe) -> DiffusersExecutor:
    executor = DiffusersExecutor(DEFAULT_WORKER_CONFIG)
    executor._pipe = pipe  # type: ignore[assignment]
    executor._model_name = "org/sd"
    executor._device = "cpu"
    monkeypatch.setattr(executor, "_ensure_pipeline", lambda spec: None)
    monkeypatch.setattr(
        executor,
        "_encode_and_combine_prompts",
        lambda prompts, system, negative, num: (list(prompts), None, None, None),
    )
    return executor


def test_merged_children_share_one_pipeline_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    uploaded: list[str] = []
    monkeypatch.setattr(
        mod,
        "maybe_upload_artifacts",
        lambda ref, out_dir, logger=None: uploaded.append(ref.task_id),
    )
    task = make_worker_task_message(
        _spec(["parent"]),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[_child("tsk-c1", ["a", "b"]), _child("tsk-c2", ["c"])],
    )

    result = executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))
    child1 = result.children["tsk-c1"]
    child2 = result.children["tsk-c2"]
    assert isinstance(child1, DiffusionResult)
    assert isinstance(child2, DiffusionResult)

    assert pipe.batch_sizes == [4]
    assert [i.path for i in result.images] == ["images/image_0.png"]
    assert [i.path for i in child1.images] == [
        "images/image_0.png",
        "images/image_1.png",
    ]
    assert [i.path for i in child2.images] == ["images/image_0.png"]

    def shade(task_id: str, idx: int) -> int:
        path = tmp_path / task_id / "artifacts" / "images" / f"image_{idx}.png"
        with Image.open(path) as image:
            pixel = image.getpixel((0, 0))
        assert isinstance(pixel, int)
        return pixel

    assert shade("tsk-parent", 0) == 0
    assert [shade("tsk-c1", 0), shade("tsk-c1", 1)] == [1, 2]
    assert shade("tsk-c2", 0) == 3
    assert uploaded == ["tsk-c1", "tsk-c2", "tsk-parent"]


def test_unmerged_task_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    task = make_worker_task_message(
        _spec(["x", "y"]), task_type=TaskType.DIFFUSION, task_id="tsk-solo"
    )

    result = executor.run(task, tmp_path / "tsk-solo", RunControl(task.task_id))

    assert pipe.batch_sizes == [2]
    assert [i.path for i in result.images] == [
        "images/image_0.png",
        "images/image_1.png",
    ]
    assert result.children == {}


def test_merged_child_of_another_type_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(monkeypatch, _FakePipe())
    echo_child = MergedChildTaskStrict(
        task_id="tsk-echo",
        owner_id="usr-other",
        workflow_id="wfl-other",
        spec=EchoSpecStrict(taskType=TaskType.ECHO),
    )
    task = make_worker_task_message(
        _spec(["p"]),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[echo_child],
    )
    with pytest.raises(ExecutionError, match="must be diffusion"):
        executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))
