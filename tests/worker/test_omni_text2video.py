"""omni_text2video: one mp4 per prompt, matched to its prompt by request id.

The engine yields outputs in completion order and tags request ids
``"{prompt_index}_{uuid}"``. Driven through the public ``run`` with a mocked
model; the encoder check runs against the installed vllm-omni.
"""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from shared.schemas.result import InferenceItem, InferenceResult, OmniText2VideoResult
from shared.tasks import MergedChildTaskStrict
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.omni import OmniText2ImageSpecStrict, OmniText2VideoSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors import omni_executor_base as base_mod
from worker.executors import omni_text2video_executor as mod
from worker.executors.base_executor import ExecutionError, RunControl
from worker.executors.omni_text2video_executor import OmniText2VideoExecutor

from .factories import DEFAULT_WORKER_CONFIG, make_worker_task_message

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm_omni") is None, reason="vllm-omni not installed"
)


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


def _request_output(frames: Any, **multimodal: Any) -> mod.OmniRequestOutput:
    return mod.OmniRequestOutput(
        request_id="0_a", images=[frames], _multimodal_output=multimodal
    )


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
    monkeypatch.setattr(executor, "_omni", fake)
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    monkeypatch.setattr(
        mod, "_encode_mp4", lambda output, cfg: f"mp4:{output.request_id}".encode()
    )
    return executor.run(task, tmp_path, RunControl(task.task_id)), fake


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

    executor._ensure_omni(_spec(["p"], task_type="t2va").model_dump(by_alias=True))
    assert len(created) == 3
    assert created[2].init_kwargs == {"model": "org/video", "task_type": "t2va"}


def test_encoder_writes_mp4_from_installed_vllm_omni() -> None:
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    data = mod._encode_mp4(_request_output(frames), 8)

    assert data[4:8] == b"ftyp"


def test_missing_num_frames_fails_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={"type": "list", "items": ["p0"]},
    )
    with pytest.raises(
        ExecutionError, match="invalid omni_text2video spec.omni: num_frames"
    ):
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
    output = _request_output(frames, fps=24)

    mod._encode_mp4(output, 8)
    assert seen["fps"] == 8

    mod._encode_mp4(output, None)
    assert seen["fps"] == 24


def test_h3_style_request_sends_extra_args_and_no_default_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(
        ["a lantern festival"],
        height=384,
        width=672,
        num_frames=124,
        extra_args={"aspect_ratio": "16:9"},
    )
    _, fake = _run(tmp_path, monkeypatch, spec, [_output("0_a")])

    ((_, params),) = fake.calls
    assert params.extra_args == {"aspect_ratio": "16:9"}
    assert params.fps is None
    assert (params.height, params.width, params.num_frames) == (384, 672, 124)


def test_extra_args_must_be_a_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(["p0"], extra_args="16:9")
    with pytest.raises(
        ExecutionError, match="invalid omni_text2video spec.omni: extra_args"
    ):
        _run(tmp_path, monkeypatch, spec, [_output("0_a")])


@pytest.mark.parametrize(
    ("omni", "field"),
    [
        ({"num_frames": 0}, "num_frames"),
        ({"num_frames": 80.5}, "num_frames"),
        ({"num_frames": True}, "num_frames"),
        ({"num_frames": "81"}, "num_frames"),
        ({"num_frames": 81, "height": 480.0}, "height"),
        ({"num_frames": 81, "seed": False}, "seed"),
        ({"num_frames": 81, "guidance_scale": True}, "guidance_scale"),
        ({"num_frames": 81, "negative_prompt": 1}, "negative_prompt"),
    ],
)
def test_sampling_options_reject_casts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, omni: dict[str, Any], field: str
) -> None:
    spec = OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={"type": "list", "items": ["p0"]},
        omni=omni,
    )
    with pytest.raises(ExecutionError, match=f"spec.omni: {field}"):
        _run(tmp_path, monkeypatch, spec, [_output("0_a")])


def test_invalid_sampling_params_fail_before_the_engine_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = OmniText2VideoSpecStrict(
        taskType=TaskType.OMNI_TEXT2VIDEO,
        model=ModelConfig(source=ModelSource(identifier="org/video")),
        data={"type": "list", "items": ["p0"]},
        omni={"extra_args": "16:9"},
    )
    task = make_worker_task_message(
        spec, task_type=TaskType.OMNI_TEXT2VIDEO, task_id="tsk-video"
    )
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)

    def _load(spec_dict: Any) -> None:
        raise AssertionError("engine loaded before sampling params were validated")

    monkeypatch.setattr(executor, "_ensure_omni", _load)
    with pytest.raises(
        ExecutionError, match="invalid omni_text2video spec.omni: num_frames"
    ):
        executor.run(task, tmp_path, RunControl(task.task_id))


@pytest.mark.parametrize(
    "quantization", ["fp8", {"method": "fp8", "ignored_layers": []}]
)
def test_quantization_reaches_omni_unchanged(
    monkeypatch: pytest.MonkeyPatch, quantization: Any
) -> None:
    created: list[_FakeOmni] = []

    def _factory(**kwargs: Any) -> _FakeOmni:
        created.append(_FakeOmni([], **kwargs))
        return created[-1]

    monkeypatch.setattr(mod, "Omni", _factory)
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    executor._ensure_omni(
        _spec(["p"], quantization=quantization).model_dump(by_alias=True)
    )

    assert created[0].init_kwargs["quantization_config"] == quantization


def test_quantization_must_be_a_name_or_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "Omni", lambda **kwargs: _FakeOmni([], **kwargs))
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    with pytest.raises(
        ExecutionError, match="invalid omni weight options: quantization"
    ):
        executor._ensure_omni(_spec(["p"], quantization=8).model_dump(by_alias=True))


def test_encoder_muxes_model_audio_into_the_mp4() -> None:
    av = pytest.importorskip("av")
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    audio = np.zeros(22050, dtype=np.float32)
    output = _request_output(frames, audio=audio, audio_sample_rate=44100)

    data = mod._encode_mp4(output, 8)

    with av.open(io.BytesIO(data)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].rate == 44100


def _child(task_id: str, items: list[str]) -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id=task_id,
        owner_id="usr-other",
        workflow_id="wfl-other",
        spec=_spec(items),
    )


def test_merged_children_share_one_generate_and_keep_their_own_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent with two merged children: every prompt goes to one generate
    call, and each task gets only its own videos, in its own output dir."""
    fake = _FakeOmni([_output("2_c"), _output("0_a"), _output("3_d"), _output("1_b")])
    task = make_worker_task_message(
        _spec(["parent"]),
        task_type=TaskType.OMNI_TEXT2VIDEO,
        task_id="tsk-parent",
        merged_children=[
            _child("tsk-child1", ["c1a", "c1b"]),
            _child("tsk-child2", ["c2"]),
        ],
    )
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    executor._model_name = "org/video"
    monkeypatch.setattr(executor, "_omni", fake)
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    monkeypatch.setattr(
        mod, "_encode_mp4", lambda output, cfg: f"mp4:{output.request_id}".encode()
    )
    uploaded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        base_mod,
        "maybe_upload_artifacts",
        lambda ref, out_dir, logger=None: uploaded.append((ref.task_id, out_dir.name)),
    )

    out_dir = tmp_path / "tsk-parent"
    result = executor.run(task, out_dir, RunControl(task.task_id))
    child1 = result.children["tsk-child1"]
    child2 = result.children["tsk-child2"]
    assert isinstance(result, OmniText2VideoResult)
    assert isinstance(child1, OmniText2VideoResult)
    assert isinstance(child2, OmniText2VideoResult)

    ((requests, _),) = fake.calls
    assert [r["prompt"] for r in requests] == ["parent", "c1a", "c1b", "c2"]
    assert [i.prompt for i in result.items] == ["parent"]
    assert (out_dir / "artifacts" / "generated_video.mp4").read_bytes() == b"mp4:0_a"
    assert [i.prompt for i in child1.items] == ["c1a", "c1b"]
    child1_dir = tmp_path / "tsk-child1" / "artifacts"
    assert (child1_dir / "generated_video_1.mp4").read_bytes() == b"mp4:1_b"
    assert (child1_dir / "generated_video_2.mp4").read_bytes() == b"mp4:2_c"
    child2_dir = tmp_path / "tsk-child2" / "artifacts"
    assert [i.prompt for i in child2.items] == ["c2"]
    assert (child2_dir / "generated_video.mp4").read_bytes() == b"mp4:3_d"
    assert ("tsk-child1", "tsk-child1") in uploaded
    assert ("tsk-child2", "tsk-child2") in uploaded


def test_merged_child_of_another_type_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_child = MergedChildTaskStrict(
        task_id="tsk-image",
        owner_id="usr-other",
        workflow_id="wfl-other",
        spec=OmniText2ImageSpecStrict(
            taskType=TaskType.OMNI_TEXT2IMAGE,
            model=ModelConfig(source=ModelSource(identifier="org/video")),
            data={"type": "list", "items": ["x"]},
        ),
    )
    task = make_worker_task_message(
        _spec(["p"]),
        task_type=TaskType.OMNI_TEXT2VIDEO,
        task_id="tsk-parent",
        merged_children=[image_child],
    )
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    monkeypatch.setattr(executor, "_ensure_omni", lambda spec_dict: None)
    with pytest.raises(ExecutionError, match="must be omni_text2video"):
        executor.run(task, tmp_path / "tsk-parent", RunControl(task.task_id))


def test_execution_flags_reach_omni_and_key_its_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeOmni] = []

    def _factory(**kwargs: Any) -> _FakeOmni:
        created.append(_FakeOmni([], **kwargs))
        return created[-1]

    monkeypatch.setattr(mod, "Omni", _factory)
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    flags = {
        "step_execution": True,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cache_backend": "cache_dit",
        "cache_config": {"Fn_compute_blocks": 1},
    }
    executor._ensure_omni(_spec(["p"], **flags).model_dump(by_alias=True))
    executor._ensure_omni(_spec(["p"], **flags).model_dump(by_alias=True))
    assert len(created) == 1
    assert created[0].init_kwargs == {"model": "org/video", **flags}

    executor._ensure_omni(_spec(["p"], step_execution=False).model_dump(by_alias=True))
    assert len(created) == 2
    assert created[1].init_kwargs == {"model": "org/video", "step_execution": False}


_EXEC = "invalid omni execution options: "
_WEIGHT = "invalid omni weight options: "


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"ulysses_degree": 0}, _EXEC + "ulysses_degree"),
        ({"ulysses_degree": 1.5}, _EXEC + "ulysses_degree"),
        ({"ulysses_degree": 2.0}, _EXEC + "ulysses_degree"),
        ({"ring_degree": True}, _EXEC + "ring_degree"),
        ({"ring_degree": "2"}, _EXEC + "ring_degree"),
        ({"step_execution": "yes"}, _EXEC + "step_execution"),
        ({"cache_backend": ""}, _EXEC + "cache_backend"),
        ({"cache_backend": "tea_cache", "cache_config": "x"}, _EXEC + "cache_config"),
        ({"cache_config": {"a": 1}}, _EXEC + ".*cache_config needs cache_backend"),
        ({"enable_cpu_offload": "yes"}, _WEIGHT + "enable_cpu_offload"),
        ({"enable_layerwise_offload": 1}, _WEIGHT + "enable_layerwise_offload"),
        ({"task_type": 3}, "task_type must be a string"),
    ],
)
def test_invalid_engine_options_fail(
    monkeypatch: pytest.MonkeyPatch, flags: dict[str, Any], message: str
) -> None:
    monkeypatch.setattr(mod, "Omni", lambda **kwargs: _FakeOmni([], **kwargs))
    executor = OmniText2VideoExecutor(DEFAULT_WORKER_CONFIG)
    with pytest.raises(ExecutionError, match=message):
        executor._ensure_omni(_spec(["p"], **flags).model_dump(by_alias=True))
