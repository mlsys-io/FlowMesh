"""Omni executor for video generation via vllm_omni."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, PositiveInt

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import OmniResult, OmniText2VideoResult, OmniVideoItem
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2VideoSpecStrict
from shared.tasks.task_type import TaskType
from worker.config import WorkerConfig

from .base_executor import ExecutionError
from .omni_executor_base import (
    _HAS_OMNI,
    Omni,
    OmniExecutorBase,
    OmniMember,
    OmniRequestOutput,
    prompt_index_from_request_id,
    validate_options,
)

try:
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
    from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
    from vllm_omni.outputs.output_metadata import DiffusionPayloadValue

    _HAS_OMNI_VIDEO = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
        from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
        from vllm_omni.outputs.output_metadata import DiffusionPayloadValue
    else:
        OmniOpenAIServingVideo = object
        OmniDiffusionSamplingParams = object
        OmniTextPrompt = object
        DiffusionPayloadValue = object
        _encode_video_bytes = None
    _HAS_OMNI_VIDEO = False

logger = logging.getLogger(__name__)
EXECUTOR_NAME = "omni_text2video"

_DEFAULT_FPS = 16


class VideoSamplingOptions(BaseModel):
    """``spec.omni`` keys that shape each rendered video."""

    model_config = ConfigDict(strict=True)

    num_frames: PositiveInt
    height: PositiveInt | None = None
    width: PositiveInt | None = None
    num_inference_steps: PositiveInt | None = None
    seed: int | None = None
    fps: PositiveInt | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    extra_args: dict[str, Any] | None = None


class OmniText2VideoExecutor(OmniExecutorBase):
    """Generate videos using vllm_omni.Omni.

    All prompts of a task and of its merged children go to one
    ``Omni.generate`` call with the same sampling parameters; each prompt
    yields one mp4 under its task's ``artifacts/``.
    """

    name = EXECUTOR_NAME
    supported_task_types = frozenset({TaskType.OMNI_TEXT2VIDEO})
    _TASK_SPEC_TYPE = OmniText2VideoSpecStrict

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return _HAS_OMNI and _HAS_OMNI_VIDEO

    def prepare(self) -> None:
        if not _HAS_OMNI or not _HAS_OMNI_VIDEO:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2video executor."
            )

    def _run_inner(
        self,
        task_id: str,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniResult:
        return self._run_members([OmniMember(task_id, spec, out_dir)])[0]

    def _run_members(self, members: list[OmniMember]) -> list[OmniResult]:
        """Render the prompts of every member in one ``Omni.generate`` call;
        each member's videos go to its own ``artifacts/``."""
        task_id = members[0].task_id
        spec_dict = members[0].spec.model_dump(by_alias=True)
        cfg = self.omni_cfg(spec_dict)
        options = _sampling_options(cfg)
        prompts_by_member = [
            self._collect_text_inputs(member.spec, member.task_id) for member in members
        ]
        prompts = [p for member_prompts in prompts_by_member for p in member_prompts]

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)

        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={
                "task_id": task_id,
                "prompt_count": len(prompts),
                "merged_tasks": len(members),
            },
        ):
            videos = self._generate_videos(prompts, options)

        results: list[OmniResult] = []
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task_id, "item_count": len(prompts)},
        ):
            offset = 0
            for member, member_prompts in zip(members, prompts_by_member):
                member_videos = videos[offset : offset + len(member_prompts)]
                offset += len(member_prompts)
                results.append(
                    self._save_videos(
                        cfg, member.out_dir, member_prompts, member_videos
                    )
                )
        return results

    def _save_videos(
        self,
        cfg: dict[str, Any],
        out_dir: Path,
        prompts: list[str],
        videos: list[bytes],
    ) -> OmniText2VideoResult:
        """Write one task's mp4s under ``out_dir/artifacts`` and describe them."""
        artifacts_dir = out_dir / "artifacts"
        items: list[OmniVideoItem] = []
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
                    video=ArtifactRef(path=self.relative_to(save_path, artifacts_dir)),
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
        # vLLM-Omni's checkpoint partition, e.g. MiniMax-H3 "t2va" loads only
        # FL2VA. Not shared: other executors read spec.omni.task_type differently.
        task_type = cfg.get("task_type") or None
        if task_type is not None and not isinstance(task_type, str):
            raise ExecutionError(
                "omni_text2video spec.omni.task_type must be a string."
            )
        new_spec = (*self._build_omni_spec(model_name, cfg), task_type)
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
        if task_type:
            init_kwargs["task_type"] = task_type
        self._omni = Omni(**init_kwargs)
        self._model_name = model_name
        self._omni_spec = new_spec

    # ── generation ───────────────────────────────────────────────────────

    def _generate_videos(
        self, prompts: list[str], options: VideoSamplingOptions
    ) -> list[bytes]:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        requests: list[OmniTextPrompt] = []
        for prompt in prompts:
            request: OmniTextPrompt = {"prompt": prompt}
            if options.negative_prompt:
                request["negative_prompt"] = options.negative_prompt
            requests.append(request)
        params = _sampling_params(options)
        outputs = self._omni.generate(requests, params, use_tqdm=False)
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
        return [_encode_mp4(by_index[i], options.fps) for i in range(len(prompts))]


# ── config / output helpers ──────────────────────────────────────────────────


def _sampling_options(cfg: dict[str, Any]) -> VideoSamplingOptions:
    return validate_options(VideoSamplingOptions, cfg, "omni_text2video spec.omni")


def _sampling_params(options: VideoSamplingOptions) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        **options.model_dump(exclude_none=True, exclude={"negative_prompt"})
    )


def _prompt_index(request_id: str, prompt_count: int) -> int:
    """Return the prompt index encoded as the request id's leading field."""
    idx = prompt_index_from_request_id(request_id)
    if idx is None or idx >= prompt_count:
        raise ExecutionError(f"request id {request_id!r} has no valid prompt index.")
    return idx


def _encode_mp4(output: OmniRequestOutput, fps: int | None) -> bytes:
    """Encode one request's video (and audio, when the model emits it) as mp4."""
    serving = OmniOpenAIServingVideo
    multimodal = output.multimodal_output or {}
    images: list[DiffusionPayloadValue] = list(output.images)
    videos = serving._normalize_video_outputs(images or multimodal.get("video"))
    if not videos:
        raise ExecutionError("Omni video generation returned no video.")
    return _encode_video_bytes(
        videos[0],
        fps=fps or serving._resolve_fps(output) or _DEFAULT_FPS,
        audio=serving._extract_audio_outputs(output, expected_count=len(videos))[0],
        audio_sample_rate=serving._extract_audio_sample_rate_from_result(output),
    )
