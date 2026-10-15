"""Omni executors run a task's merged children in the same dispatch.

text2image renders every member's prompts in one batched ``Omni.generate``
call and matches images back by request id; the other Omni executors run the
members one after another on the loaded model. Each member's outputs land in
its own output dir and its result rides on ``result.children``.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from shared.schemas.result import OmniText2ImageResult, OmniText2SpeechResult
from shared.tasks import MergedChildTaskStrict
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.omni import OmniText2ImageSpecStrict, OmniText2SpeechSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors import omni_executor_base as base_mod
from worker.executors.base_executor import ExecutionError, RunControl
from worker.executors.omni_text2image_executor import OmniText2ImageExecutor
from worker.executors.omni_text2speech_executor import OmniText2SpeechExecutor

from .factories import DEFAULT_WORKER_CONFIG, make_worker_task_message

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm_omni") is None, reason="vllm-omni not installed"
)


class _FakeOmni:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.calls: list[Any] = []

    def generate(self, prompts: Any, **kwargs: Any) -> list[Any]:
        self.calls.append(prompts)
        return list(self.outputs)


def _image_output(request_id: str, shade: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id, images=[Image.new("L", (2, 2), color=shade)]
    )


def _image_spec(items: list[str]) -> OmniText2ImageSpecStrict:
    return OmniText2ImageSpecStrict(
        taskType=TaskType.OMNI_TEXT2IMAGE,
        model=ModelConfig(source=ModelSource(identifier="org/image")),
        data={"type": "list", "items": items},
    )


def _speech_spec(items: list[str]) -> OmniText2SpeechSpecStrict:
    return OmniText2SpeechSpecStrict(
        taskType=TaskType.OMNI_TEXT2SPEECH,
        model=ModelConfig(source=ModelSource(identifier="org/tts")),
        data={"type": "list", "items": items},
    )


def _child(
    task_id: str, spec: OmniText2ImageSpecStrict | OmniText2SpeechSpecStrict
) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id, owner_id="usr-other", workflow_id="wfl-other", spec=spec
    )


@pytest.fixture
def uploaded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        base_mod,
        "maybe_upload_artifacts",
        lambda ref, out_dir, logger=None: calls.append((ref.task_id, out_dir.name)),
    )
    return calls


def _image_executor(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeOmni
) -> OmniText2ImageExecutor:
    executor = OmniText2ImageExecutor(DEFAULT_WORKER_CONFIG)
    executor._model_name = "org/image"
    monkeypatch.setattr(executor, "_omni", fake)
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    return executor


def test_image_children_share_one_generate_matched_by_request_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uploaded: list[tuple[str, str]],
) -> None:
    fake = _FakeOmni(
        [_image_output("2_c", 30), _image_output("0_a", 10), _image_output("1_b", 20)]
    )
    executor = _image_executor(monkeypatch, fake)
    task = make_worker_task_message(
        _image_spec(["parent"]),
        task_type=TaskType.OMNI_TEXT2IMAGE,
        task_id="tsk-parent",
        merged_children=[_child("tsk-child", _image_spec(["c1", "c2"]))],
    )

    result = executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))

    assert fake.calls == [["parent", "c1", "c2"]]
    child = result.children["tsk-child"]
    assert isinstance(result, OmniText2ImageResult)
    assert isinstance(child, OmniText2ImageResult)
    assert [i.prompt for i in result.items] == ["parent"]
    assert [i.prompt for i in child.items] == ["c1", "c2"]
    parent_png = tmp_path / "tsk-parent" / "artifacts" / "generated_image.png"
    child_dir = tmp_path / "tsk-child" / "artifacts"
    assert Image.open(parent_png).getpixel((0, 0)) == 10
    assert Image.open(child_dir / "generated_image_1.png").getpixel((0, 0)) == 20
    assert Image.open(child_dir / "generated_image_2.png").getpixel((0, 0)) == 30
    assert ("tsk-child", "tsk-child") in uploaded


@pytest.mark.parametrize(
    "request_ids", [["0_a", "0_b"], ["0_a", "5_b"], ["0_a", "x_b"], ["0_a"]]
)
def test_image_batch_with_unmatched_request_ids_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_ids: list[str]
) -> None:
    fake = _FakeOmni([_image_output(rid, 0) for rid in request_ids])
    executor = _image_executor(monkeypatch, fake)
    task = make_worker_task_message(
        _image_spec(["p0", "p1"]), task_type=TaskType.OMNI_TEXT2IMAGE
    )
    with pytest.raises(ExecutionError, match="Omni image batch"):
        executor.run(task, tmp_path / "tsk", RunControl(task.task_id))


def test_speech_children_run_in_turn_on_the_loaded_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uploaded: list[tuple[str, str]],
) -> None:
    executor = OmniText2SpeechExecutor(DEFAULT_WORKER_CONFIG)
    executor._model_name = "org/tts"
    loads: list[dict[str, Any]] = []
    monkeypatch.setattr(executor, "_ensure_omni", loads.append)
    monkeypatch.setattr(
        executor,
        "_generate_single",
        lambda text, spec_dict: f"wav:{text}".encode(),
    )
    task = make_worker_task_message(
        _speech_spec(["hello"]),
        task_type=TaskType.OMNI_TEXT2SPEECH,
        task_id="tsk-parent",
        merged_children=[_child("tsk-child", _speech_spec(["bye"]))],
    )

    result = executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))

    child = result.children["tsk-child"]
    assert isinstance(result, OmniText2SpeechResult)
    assert isinstance(child, OmniText2SpeechResult)
    assert [i.text for i in result.items] == ["hello"]
    assert [i.text for i in child.items] == ["bye"]
    parent_wav = tmp_path / "tsk-parent" / "artifacts" / "generated_tts.wav"
    child_wav = tmp_path / "tsk-child" / "artifacts" / "generated_tts.wav"
    assert parent_wav.read_bytes() == b"wav:hello"
    assert child_wav.read_bytes() == b"wav:bye"
    assert len(loads) == 2
    assert ("tsk-child", "tsk-child") in uploaded
