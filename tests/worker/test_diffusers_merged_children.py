"""DiffusersExecutor runs a diffusion task with its merged children: each
task's images, uploads and asset owner, per-call image caps, and seeded
reproducibility. Driven through the public ``run`` with a fake pipeline."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from shared.schemas.result import DiffusionResult
from shared.tasks import MergedChildTaskStrict
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import DiffusionSpecStrict, EchoSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors import diffusers_executor as mod
from worker.executors.base_executor import ExecutionError, RunControl
from worker.executors.diffusers_executor import DiffusersExecutor

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


class _FakePipe:
    """Returns one solid image per embedding. Unseeded, an image is shaded by
    its position across all calls; seeded, by a draw from its generator."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._count = 0

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        prompts = list(kwargs["prompt_embeds"])
        self.calls.append(prompts)
        generators = kwargs["generator"]
        shades: list[int] = []
        for i in range(len(prompts)):
            if generators is None:
                shades.append(self._count)
            else:
                draw = mod.torch.randint(0, 256, (1,), generator=generators[i])
                shades.append(int(draw.item()))
            self._count += 1
        return SimpleNamespace(
            images=[Image.new("L", (4, 4), color=shade) for shade in shades]
        )


def _spec(items: list[str], **inference: Any) -> DiffusionSpecStrict:
    return DiffusionSpecStrict(
        taskType=TaskType.DIFFUSION,
        model=ModelConfig(source=ModelSource(identifier="org/sd"), diffusers={}),
        data={"type": "list", "items": items},
        inference={"num_inference_steps": 2, **inference},
    )


def _child(task_id: str, items: list[str], **inference: Any) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id="usr-other",
        workflow_id="wfl-other",
        spec=_spec(items, **inference),
    )


def _executor(monkeypatch: pytest.MonkeyPatch, pipe: _FakePipe) -> DiffusersExecutor:
    """An executor on ``pipe`` whose encoder yields each prompt once per
    image, grouped per prompt like the real one."""
    executor = DiffusersExecutor(DEFAULT_WORKER_CONFIG)
    executor._pipe = pipe
    executor._model_name = "org/sd"
    executor._device = "cpu"
    monkeypatch.setattr(executor, "_ensure_pipeline", lambda spec: None)
    monkeypatch.setattr(
        executor,
        "_encode_and_combine_prompts",
        lambda prompts, system, negative, num: (
            [p for p in prompts for _ in range(num)],
            None,
            None,
            None,
        ),
    )
    return executor


def _shades(out_dir: Path) -> list[int]:
    shades: list[int] = []
    for path in sorted((out_dir / "artifacts" / "images").glob("image_*.png")):
        with Image.open(path) as image:
            pixel = image.getpixel((0, 0))
        assert isinstance(pixel, int)
        shades.append(pixel)
    return shades


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

    assert pipe.calls == [["parent", "a", "b", "c"]]
    assert [i.path for i in result.images] == ["images/image_0.png"]
    assert [i.path for i in child1.images] == [
        "images/image_0.png",
        "images/image_1.png",
    ]
    assert [i.path for i in child2.images] == ["images/image_0.png"]
    assert _shades(tmp_path / "tsk-parent") == [0]
    assert _shades(tmp_path / "tsk-c1") == [1, 2]
    assert _shades(tmp_path / "tsk-c2") == [3]
    assert uploaded == ["tsk-parent", "tsk-c1", "tsk-c2"]

    assets = tmp_path / "tsk-parent" / "logs" / "assets.jsonl"
    owners = [
        (row["data_id"], row["user_id"])
        for row in map(json.loads, assets.read_text().splitlines())
    ]
    assert sorted(owners) == [
        ("tsk-c1", "usr-other"),
        ("tsk-c2", "usr-other"),
        ("tsk-parent", "usr-test"),
    ]


def test_merged_children_split_images_per_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setenv("WORKER_DIFFUSION_MAX_IMAGES_PER_CALL", "8")
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    task = make_worker_task_message(
        _spec(["p"], num_images_per_prompt=2),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[_child("tsk-c1", ["a", "b"], num_images_per_prompt=2)],
    )

    executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))

    assert pipe.calls == [["p", "p", "a", "a", "b", "b"]]
    assert _shades(tmp_path / "tsk-parent") == [0, 1]
    assert _shades(tmp_path / "tsk-c1") == [2, 3, 4, 5]


def test_merged_tasks_are_packed_whole_under_the_per_call_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setenv("WORKER_DIFFUSION_MAX_IMAGES_PER_CALL", "2")
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    task = make_worker_task_message(
        _spec(["p"]),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[
            _child("tsk-c1", ["a", "b"]),
            _child("tsk-c2", ["c"]),
            _child("tsk-c3", ["d"]),
        ],
    )

    executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))

    assert pipe.calls == [["p"], ["a", "b"], ["c", "d"]]
    assert _shades(tmp_path / "tsk-parent") == [0]
    assert _shades(tmp_path / "tsk-c1") == [1, 2]
    assert _shades(tmp_path / "tsk-c2") == [3]
    assert _shades(tmp_path / "tsk-c3") == [4]


def test_task_larger_than_the_cap_runs_in_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setenv("WORKER_DIFFUSION_MAX_IMAGES_PER_CALL", "1")
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    task = make_worker_task_message(
        _spec(["x", "y", "z"]), task_type=TaskType.DIFFUSION, task_id="tsk-solo"
    )

    executor.run(task, tmp_path / "tsk-solo", RunControl(task.task_id))

    assert pipe.calls == [["x", "y", "z"]]


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "true"])
def test_invalid_per_call_cap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setenv("WORKER_DIFFUSION_MAX_IMAGES_PER_CALL", value)
    task = make_worker_task_message(
        _spec(["p"]), task_type=TaskType.DIFFUSION, task_id="tsk-solo"
    )
    with pytest.raises(ExecutionError, match="WORKER_DIFFUSION_MAX_IMAGES_PER_CALL"):
        executor.run(task, tmp_path / "tsk-solo", RunControl(task.task_id))
    assert pipe.calls == []


def test_seeded_task_renders_the_same_merged_or_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    monkeypatch.setenv("WORKER_DIFFUSION_MAX_IMAGES_PER_CALL", "2")
    alone = make_worker_task_message(
        _spec(["x"], seed=7, num_images_per_prompt=2),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-alone",
    )
    _executor(monkeypatch, _FakePipe()).run(
        alone, tmp_path / "tsk-alone", RunControl(alone.task_id)
    )
    merged = make_worker_task_message(
        _spec(["a", "b"], seed=7, num_images_per_prompt=2),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[_child("tsk-c1", ["x"], seed=7, num_images_per_prompt=2)],
    )
    pipe = _FakePipe()
    _executor(monkeypatch, pipe).run(
        merged, tmp_path / "tsk-parent", RunControl(merged.task_id)
    )

    assert len(pipe.calls) == 2
    assert _shades(tmp_path / "tsk-c1") == _shades(tmp_path / "tsk-alone")


def test_unmerged_task_renders_only_its_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    monkeypatch.setattr(mod, "maybe_upload_artifacts", lambda *a, **k: [])
    task = make_worker_task_message(
        _spec(["x", "y"]), task_type=TaskType.DIFFUSION, task_id="tsk-solo"
    )

    result = executor.run(task, tmp_path / "tsk-solo", RunControl(task.task_id))

    assert pipe.calls == [["x", "y"]]
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


def test_merged_child_with_another_system_prompt_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _FakePipe()
    executor = _executor(monkeypatch, pipe)
    task = make_worker_task_message(
        _spec(["p"], system_prompt="watercolor"),
        task_type=TaskType.DIFFUSION,
        task_id="tsk-parent",
        merged_children=[_child("tsk-c1", ["a"], system_prompt="photo")],
    )
    with pytest.raises(ExecutionError, match="different system_prompt"):
        executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))
    assert pipe.calls == []
