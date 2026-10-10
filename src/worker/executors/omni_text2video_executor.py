"""Omni executor for video generation via vllm_omni."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import OmniText2VideoResult, OmniVideoItem
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2VideoSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import to_bool, to_float, to_int

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import (
    _HAS_OMNI,
    Omni,
    OmniExecutorBase,
    OmniRequestOutput,
)

try:
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
    from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

    _HAS_OMNI_VIDEO = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
        from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
    else:
        OmniOpenAIServingVideo = object
        OmniDiffusionSamplingParams = object
        OmniTextPrompt = object
        _encode_video_bytes = None
    _HAS_OMNI_VIDEO = False

logger = logging.getLogger(__name__)
EXECUTOR_NAME = "omni_text2video"

_DEFAULT_FPS = 16
_ENGINE_FLAGS = ("enable_cpu_offload", "enable_layerwise_offload")


class OmniText2VideoExecutor(OmniExecutorBase):
    """Generate videos using vllm_omni.Omni.

    All prompts of a task go to one ``Omni.generate`` call with the same
    sampling parameters; each prompt yields one mp4 under ``artifacts/``.
    """

    name = EXECUTOR_NAME
    supported_task_types = frozenset({TaskType.OMNI_TEXT2VIDEO})
    _TASK_SPEC_TYPE = OmniText2VideoSpecStrict

    def prepare(self) -> None:
        if not _HAS_OMNI or not _HAS_OMNI_VIDEO:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2video executor."
            )

    def _run_inner(
        self,
        task: ExecutorTask,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniText2VideoResult:
        assert isinstance(spec, OmniText2VideoSpecStrict)
        prompts = self._collect_text_inputs(spec, task.task_id)
        cfg = self.omni_cfg(spec_dict)

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)

        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            videos = self._generate_videos(prompts, cfg)

        artifacts_dir = out_dir / "artifacts"
        items: list[OmniVideoItem] = []
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "item_count": len(prompts)},
        ):
            for idx, (prompt, video) in enumerate(zip(prompts, videos)):
                save_path = self.resolve_save_path(
                    cfg,
                    out_dir,
                    index=idx,
                    ext="mp4",
                    multi=len(prompts) > 1,
                    default_prefix="generated_video",
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(video)
                items.append(
                    OmniVideoItem(
                        index=idx,
                        prompt=prompt,
                        video=ArtifactRef(
                            path=self.relative_to(save_path, artifacts_dir)
                        ),
                    )
                )

        return OmniText2VideoResult(
            model=self.model_name,
            video=items[0].video if items else None,
            items=items,
        )

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        cfg = self.omni_cfg(spec_dict)
        model_name = self.resolve_model_identifier(
            spec_dict,
            cfg,
            env_keys=("OMNI_MODEL",),
            default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        engine_kwargs = _engine_kwargs(cfg)
        new_spec = (
            *self._build_omni_spec(model_name, cfg),
            tuple(sorted(engine_kwargs.items())),
        )
        if self._omni is not None:
            if self._omni_spec == new_spec:
                logger.info("Reusing existing Omni instance for model %s", model_name)
                return
            logger.info(
                "Releasing previous Omni instance for model %s (spec changed)",
                self._model_name,
            )
            self._close_omni()
        init_kwargs = self.build_omni_init_kwargs(model_name, cfg)
        init_kwargs.update(engine_kwargs)
        self._omni = Omni(**init_kwargs)
        self._model_name = model_name
        self._omni_spec = new_spec

    # ── generation ───────────────────────────────────────────────────────

    def _generate_videos(self, prompts: list[str], cfg: dict[str, Any]) -> list[bytes]:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        negative_prompt = cfg.get("negative_prompt")
        requests: list[OmniTextPrompt] = []
        for prompt in prompts:
            request: OmniTextPrompt = {"prompt": prompt}
            if negative_prompt:
                request["negative_prompt"] = str(negative_prompt)
            requests.append(request)
        outputs = self._omni.generate(requests, _sampling_params(cfg), use_tqdm=False)
        by_index: dict[int, OmniRequestOutput] = {}
        for output in outputs:
            idx = _prompt_index(str(output.request_id), len(prompts))
            if idx in by_index:
                raise ExecutionError(
                    f"Omni returned more than one video for prompt {idx}."
                )
            by_index[idx] = output
        if len(by_index) != len(prompts):
            raise ExecutionError(
                f"Omni video batch returned {len(by_index)} videos "
                f"for {len(prompts)} prompts."
            )
        return [_encode_mp4(by_index[i], cfg) for i in range(len(prompts))]


# ── config / output helpers ──────────────────────────────────────────────────


def _engine_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Engine options that change how the model is placed in memory."""
    kwargs: dict[str, Any] = {}
    for key in _ENGINE_FLAGS:
        if cfg.get(key) is not None:
            kwargs[key] = to_bool(cfg.get(key), default=False)
    quantization = cfg.get("quantization")
    if quantization:
        kwargs["quantization_config"] = str(quantization)
    return kwargs


def _sampling_params(cfg: dict[str, Any]) -> OmniDiffusionSamplingParams:
    num_frames = to_int(cfg.get("num_frames"))
    if num_frames is None or num_frames < 1:
        raise ExecutionError("omni_text2video requires spec.omni.num_frames.")
    params: dict[str, Any] = {
        "fps": to_int(cfg.get("fps")) or _DEFAULT_FPS,
        "num_frames": num_frames,
    }
    for key in ("height", "width", "num_inference_steps", "seed"):
        value = to_int(cfg.get(key))
        if value is not None:
            params[key] = value
    guidance_scale = to_float(cfg.get("guidance_scale"))
    if guidance_scale is not None:
        params["guidance_scale"] = guidance_scale
    return OmniDiffusionSamplingParams(**params)


def _prompt_index(request_id: str, prompt_count: int) -> int:
    """Return the prompt index encoded as the request id's leading field."""
    head = request_id.split("_", 1)[0]
    if not head.isdigit() or int(head) >= prompt_count:
        raise ExecutionError(f"request id {request_id!r} has no valid prompt index.")
    return int(head)


def _encode_mp4(output: OmniRequestOutput, cfg: dict[str, Any]) -> bytes:
    """Encode one request's video (and audio, when the model emits it) as mp4."""
    serving = OmniOpenAIServingVideo
    target = getattr(output, "request_output", None) or output
    multimodal = getattr(target, "multimodal_output", None) or {}
    videos = serving._normalize_video_outputs(
        getattr(target, "images", None) or multimodal.get("video")
    )
    if not videos:
        raise ExecutionError("Omni video generation returned no video.")
    fps = to_int(cfg.get("fps")) or serving._resolve_fps(target) or _DEFAULT_FPS
    return _encode_video_bytes(
        videos[0],
        fps=fps,
        audio=serving._extract_audio_outputs(target, expected_count=len(videos))[0],
        audio_sample_rate=serving._extract_audio_sample_rate_from_result(target),
    )
