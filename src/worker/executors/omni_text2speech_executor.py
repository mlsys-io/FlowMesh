"""Omni executor for text-to-speech via vllm_omni."""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2SpeechSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import to_int

from .base_executor import ExecutionError, ExecutorTask
from .omni import qwen3_tts
from .omni_executor_base import (
    _HAS_OMNI,
    OmniExecutorBase,
    OmniRequestOutput,
    OmniResult,
    extract_audio_from_mm,
    extract_multimodal_output,
    save_audio,
)

logger = logging.getLogger(__name__)
EXECUTOR_NAME = "omni_text2speech"


class OmniText2SpeechResult(OmniResult):
    executor: str = EXECUTOR_NAME
    mode: str = "tts"
    audio: ArtifactRef | None
    sample_rate: int
    storyboard: dict[str, Any] | None = None


class OmniText2SpeechExecutor(OmniExecutorBase):
    """Generate speech audio using vllm_omni.Omni."""

    name = EXECUTOR_NAME
    supported_task_types = frozenset({TaskType.OMNI_TEXT2SPEECH})
    _TASK_SPEC_TYPE = OmniText2SpeechSpecStrict

    def prepare(self) -> None:
        if not _HAS_OMNI:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2speech executor."
            )

    def _run_inner(
        self,
        task: ExecutorTask,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniText2SpeechResult:
        assert isinstance(spec, OmniText2SpeechSpecStrict)
        texts = self._collect_text_inputs(spec, task.task_id)

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(texts)},
        ):
            self._ensure_omni(spec_dict)
        cfg = self.omni_cfg(spec_dict, "omni:tts", "omni_text2speech")
        output_format = str(cfg.get("output_format") or "").strip().lower() or "wav"
        sample_rate = to_int(cfg.get("sample_rate"), default=24000)

        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(texts)},
        ):
            audio_objects = [
                self._generate_single(t, spec_dict=spec_dict) for t in texts
            ]
        artifacts_dir = out_dir / "artifacts"
        items: list[dict[str, Any]] = []
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "item_count": len(texts)},
        ):
            for idx, (text, audio_obj) in enumerate(zip(texts, audio_objects)):
                save_path = self.resolve_save_path(
                    cfg,
                    out_dir,
                    index=idx,
                    ext=output_format,
                    multi=len(texts) > 1,
                    default_prefix="generated_tts",
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(audio_obj, save_path, sample_rate=sample_rate)
                items.append(
                    {
                        "index": idx,
                        "text": text,
                        "audio": ArtifactRef(
                            path=self.relative_to(save_path, artifacts_dir)
                        ),
                    }
                )

        return OmniText2SpeechResult(
            model=self._model_name,
            items=items,
            audio=items[0]["audio"] if items else None,
            sample_rate=sample_rate,
            storyboard=spec_dict.get("storyboard"),
        )

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        from vllm_omni.entrypoints.omni import Omni

        cfg = self.omni_cfg(spec_dict, "omni:tts", "omni_text2speech")
        model_name = self.resolve_model_identifier(
            spec_dict,
            cfg,
            env_keys=("OMNI_TTS_MODEL",),
            default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        )
        if qwen3_tts.is_qwen3_tts(model_name) and cfg.get("async_chunk") is None:
            cfg = {**cfg, "async_chunk": False}
        new_spec = self._build_omni_spec(model_name, cfg)
        if self._omni is not None:
            if self._omni_spec == new_spec:
                logger.info("Reusing existing Omni instance for model %s", model_name)
                return
            logger.info(
                "Releasing previous Omni instance for model %s (spec changed)",
                self._model_name,
            )
            self._close_omni()
        self._omni = Omni(**self.build_omni_init_kwargs(model_name, cfg))
        self._model_name = model_name
        self._omni_spec = new_spec

    # ── generation ───────────────────────────────────────────────────────

    def _generate_single(self, text: str, spec_dict: dict[str, Any]) -> Any:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        model_name = str(self._model_name or "")
        prompt = self._build_tts_prompt(text, spec_dict=spec_dict)
        sampling_params = (
            qwen3_tts.build_sampling_params(self._omni)
            if qwen3_tts.is_qwen3_tts(model_name)
            else None
        )
        outputs = self._omni.generate(
            prompt, sampling_params_list=sampling_params, use_tqdm=False
        )
        audio = _extract_first_audio(outputs)
        if audio is None:
            raise ExecutionError(
                "Omni TTS generation returned no audio "
                f"({_summarize_omni_outputs(outputs)})."
            )
        return audio

    def _build_tts_prompt(self, text: str, spec_dict: dict[str, Any]) -> Any:
        model_name = str(self._model_name or "")
        if not qwen3_tts.is_qwen3_tts(model_name):
            # TODO(omni-tts): other vllm_omni TTS families (higgs_audio, ming_tts,
            # moss_tts, voxcpm2, glm_tts, cosyvoice3, …) each need their own prompt
            # / additional_information builder — see vllm_omni
            # entrypoints/openai/serving_speech.py adapters. Until those exist, pass
            # the raw text, which only drives models that accept a string prompt.
            return text
        cfg = self.omni_cfg(spec_dict, "omni:tts", "omni_text2speech")
        data = spec_dict.get("data") if isinstance(spec_dict.get("data"), dict) else {}
        if not isinstance(data, dict):
            data = {}
        return qwen3_tts.build_prompt(
            model_name=model_name, text=text, cfg=cfg, data=data
        )


# ── TTS-specific helpers ─────────────────────────────────────────────────────


def _extract_first_audio(outputs: Iterable[OmniRequestOutput]) -> Any:
    for stage in outputs:
        mm = extract_multimodal_output(stage)
        audio = extract_audio_from_mm(mm)
        if audio is not None:
            return audio
    return None


def _summarize_omni_outputs(outputs: Iterable[OmniRequestOutput]) -> str:
    parts: list[str] = []
    for index, stage in enumerate(outputs):
        stage_bits = [
            f"stage[{index}]={type(stage).__name__}",
            f"stage_id={stage.stage_id!r}",
            f"final_output_type={stage.final_output_type!r}",
            f"finished={stage.finished!r}",
        ]
        stage_bits.append(f"request_output={_summarize_request_output(stage)}")
        parts.append(", ".join(stage_bits))
    return " | ".join(parts) if parts else "outputs=[]"


def _summarize_request_output(output: OmniRequestOutput | None) -> str:
    if output is None:
        return "None"
    mm = extract_multimodal_output(output)
    outputs = output.outputs
    mm_keys = sorted(mm.keys()) if isinstance(mm, Mapping) else None
    finish_reason = None
    output_kind = None
    if outputs:
        finish_reason = getattr(outputs[0], "finish_reason", None)
        output_kind = getattr(outputs[0], "output_kind", None)
    return (
        f"ro={type(output).__name__}"
        f", finished={output.finished!r}"
        f", outputs={len(outputs) if isinstance(outputs, list) else None!r}"
        f", finish_reason={finish_reason!r}"
        f", output_kind={output_kind!r}"
        f", mm_keys={mm_keys!r}"
    )
