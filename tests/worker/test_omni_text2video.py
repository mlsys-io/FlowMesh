"""omni_text2video: one mp4 per prompt, matched to its prompt by request id.

The engine yields outputs in completion order and tags request ids
``"{prompt_index}_{uuid}"``. Driven through the public ``run`` with a mocked
model; the encoder check runs against the installed vllm-omni.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("vllm_omni", reason="vllm-omni not installed")

import numpy as np  # noqa: E402

from shared.schemas.result import InferenceItem, InferenceResult  # noqa: E402
from shared.tasks.components.model import ModelConfig, ModelSource  # noqa: E402
from shared.tasks.specs.omni import OmniText2VideoSpecStrict  # noqa: E402
from shared.tasks.task_type import TaskType  # noqa: E402
from worker.executors import omni_text2video_executor as mod  # noqa: E402
from worker.executors.base_executor import ExecutionError  # noqa: E402
from worker.executors.omni_text2video_executor import (  # noqa: E402
    OmniText2VideoExecutor,
)

from .factories import DEFAULT_WORKER_CONFIG, make_worker_task_message


class _FakeOmni:
    def __init__(self, outputs: list[Any], **init_kwargs: Any) -> None:
        self.outputs = outputs
        self.init_kwargs = init_kwargs
        self.calls: list[tuple[Any, Any]] = []

    def generate(self, prompts: Any, sampling_params: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((prompts, sampling_params))
        return list(self.outputs)

    def close(self) -> None:
        pass


def _output(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, images=None, multimodal_output={})


def _spec(items: list[str], **omni: Any) -> OmniText2VideoSpecStrict:
    return OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={"type": "list", "items": items},
        omni={"num_frames": 81, **omni},
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: OmniText2VideoSpecStrict,
    outputs: list[Any],
) -> tuple[Any, _FakeOmni]:
    fake = _FakeOmni(outputs)
    task = make_worker_task_message(
        spec, task_type=TaskType.OMNI_TEXT2VIDEO, task_id="tsk-video"
    )
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    executor._model_name = "org/video"
    executor._omni = fake  # type: ignore[assignment]
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    monkeypatch.setattr(
        mod, "_encode_mp4", lambda output, cfg: f"mp4:{output.request_id}".encode()
    )
    return executor.run(task, tmp_path), fake


def test_videos_follow_request_id_not_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(
        ["a fox in snow", "a harbor at dawn"],
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=30,
        seed=7,
        fps=16,
        guidance_scale=5.0,
        negative_prompt="blurry",
    )
    result, fake = _run(tmp_path, monkeypatch, spec, [_output("1_b"), _output("0_a")])

    assert [i.prompt for i in result.items] == ["a fox in snow", "a harbor at dawn"]
    assert [i.video.path for i in result.items] == [
        "generated_video_1.mp4",
        "generated_video_2.mp4",
    ]
    assert result.video == result.items[0].video
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "generated_video_1.mp4").read_bytes() == b"mp4:0_a"
    assert (artifacts / "generated_video_2.mp4").read_bytes() == b"mp4:1_b"

    ((requests, params),) = fake.calls
    assert requests == [
        {"prompt": "a fox in snow", "negative_prompt": "blurry"},
        {"prompt": "a harbor at dawn", "negative_prompt": "blurry"},
    ]
    assert (params.height, params.width, params.num_frames) == (480, 832, 81)
    assert (params.num_inference_steps, params.seed, params.fps) == (30, 7, 16)
    assert params.guidance_scale == 5.0


def test_prompt_comes_from_upstream_llm_via_graph_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = InferenceResult(
        items=[
            InferenceItem(
                index=0, prompt="idea", output="a fox in snow", finish_reason="stop"
            )
        ]
    )
    spec = OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={
            "type": "graph_template",
            "template": {
                "name": "video_prompt",
                "text": "Cinematic, {col0_value}",
                "columns": [{"node": "writer", "path": "items[0].output"}],
            },
        },
        omni={"num_frames": 81},
        _upstreamResults={"writer": upstream},
    )

    result, fake = _run(tmp_path, monkeypatch, spec, [_output("0_a")])

    assert [i.prompt for i in result.items] == ["Cinematic, a fox in snow"]
    assert fake.calls[0][0] == [{"prompt": "Cinematic, a fox in snow"}]


def test_missing_video_fails_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExecutionError, match="1 videos for 2 prompts"):
        _run(tmp_path, monkeypatch, _spec(["p0", "p1"]), [_output("0_a")])


def test_duplicate_video_fails_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExecutionError, match="more than one video for prompt 0"):
        _run(tmp_path, monkeypatch, _spec(["p0"]), [_output("0_a"), _output("0_b")])


def test_engine_flags_reach_omni_and_key_its_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeOmni] = []

    def _factory(**kwargs: Any) -> _FakeOmni:
        created.append(_FakeOmni([], **kwargs))
        return created[-1]

    monkeypatch.setattr(mod, "Omni", _factory)
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)

    offload = {"enable_cpu_offload": True}
    executor._ensure_omni(_spec(["p"], **offload).model_dump(by_alias=True))
    executor._ensure_omni(_spec(["p"], **offload).model_dump(by_alias=True))
    assert len(created) == 1
    assert created[0].init_kwargs == {
        "model": "org/video",
        "enable_cpu_offload": True,
    }

    executor._ensure_omni(_spec(["p"]).model_dump(by_alias=True))
    assert len(created) == 2
    assert created[1].init_kwargs == {"model": "org/video"}


def test_encoder_writes_mp4_from_installed_vllm_omni() -> None:
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    output = SimpleNamespace(request_id="0_a", images=[frames], multimodal_output={})

    data = mod._encode_mp4(output, {"fps": 8})  # type: ignore[arg-type]

    assert data[4:8] == b"ftyp"


def test_missing_num_frames_fails_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={"type": "list", "items": ["p0"]},
    )
    with pytest.raises(ExecutionError, match="requires spec.omni.num_frames"):
        _run(tmp_path, monkeypatch, spec, [_output("0_a")])


def test_configured_fps_wins_over_model_reported_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _encode(video: Any, fps: int, **kwargs: Any) -> bytes:
        seen["fps"] = fps
        return b""

    monkeypatch.setattr(mod, "_encode_video_bytes", _encode)
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    output = SimpleNamespace(
        request_id="0_a", images=[frames], multimodal_output={"fps": 24}
    )

    mod._encode_mp4(output, {"fps": 8})  # type: ignore[arg-type]
    assert seen["fps"] == 8

    mod._encode_mp4(output, {})  # type: ignore[arg-type]
    assert seen["fps"] == 24
