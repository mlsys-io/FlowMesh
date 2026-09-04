"""omni_text2general labels each result item's prompt by its request id.

The engine tags request ids ``"{prompt_index}_{uuid}"`` and can emit more than
one audio output per request, so each item's ``prompt`` is the input that
produced its audio, matched by that id. Driven through the public ``run`` with a
mocked model.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference)")

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.omni import OmniText2GeneralSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors.base_executor import ExecutionError
from worker.executors.omni_text2general_executor import OmniText2GeneralExecutor

from .factories import DEFAULT_WORKER_CONFIG, make_worker_task_message


class _FakeOmni:
    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = outputs

    def generate(self, prompts: Any, sampling_params: Any, **kwargs: Any) -> list[Any]:
        return list(self._outputs)

    def close(self) -> None:
        pass


def _audio_output(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        final_output_type="audio",
        error=None,
        multimodal_output={"audio": [0.1, -0.1], "sample_rate": 24000},
        outputs=[SimpleNamespace(text="")],
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    items: list[str],
    outputs: list[Any],
) -> Any:
    spec = OmniText2GeneralSpecStrict(
        taskType=TaskType.OMNI_TEXT2GENERAL,
        model=ModelConfig(source=ModelSource(identifier="org/omni")),
        data={"type": "list", "items": items},
        omni={"output_format": "wav"},
    )
    task = make_worker_task_message(
        spec, task_type=TaskType.OMNI_TEXT2GENERAL, task_id="tsk-omni"
    )
    executor = OmniText2GeneralExecutor(DEFAULT_WORKER_CONFIG)
    executor._model_name = "org/omni"
    executor._omni = _FakeOmni(outputs)  # type: ignore[assignment]
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    monkeypatch.setattr(
        "worker.executors.omni_text2general_executor.save_audio",
        lambda *a, **k: None,
    )
    return executor.run(task, tmp_path)


def test_items_follow_request_id_when_a_request_emits_multiple_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p0, p1 = "first prompt", "second prompt"
    # Request 0 emits two audio chunks, interleaved with request 1.
    outputs = [_audio_output("0_a"), _audio_output("1_b"), _audio_output("0_a")]

    result = _run(tmp_path, monkeypatch, [p0, p1], outputs)

    assert [item.request_id for item in result.items] == ["0_a", "1_b", "0_a"]
    assert [item.prompt for item in result.items] == [p0, p1, p0]


def test_run_fails_when_a_request_id_has_no_valid_prompt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [_audio_output("5_x")]  # index past the single prompt

    with pytest.raises(ExecutionError):
        _run(tmp_path, monkeypatch, ["only prompt"], outputs)
