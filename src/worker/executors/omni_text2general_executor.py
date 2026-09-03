"""Omni executor for narration/speech generation via Qwen3-Omni."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from vllm import SamplingParams

    _HAS_VLLM = True
except Exception:
    if TYPE_CHECKING:
        from vllm import SamplingParams
    else:
        SamplingParams = None
    _HAS_VLLM = False

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniTextPrompt
else:
    OmniTextPrompt = object

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import OmniGeneralItem, OmniText2GeneralResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2GeneralSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.parsing import to_bool, to_float, to_int, to_int_list
from worker.config import WorkerConfig

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import (
    _HAS_OMNI,
    Omni,
    OmniExecutorBase,
    OmniRequestOutput,
    RequestOutput,
    extract_audio_from_mm,
    extract_multimodal_output,
    save_audio,
)

logger = logging.getLogger(__name__)
EXECUTOR_NAME = "omni_text2general"

_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, "
    "Alibaba Group, capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech."
)


class OmniText2GeneralExecutor(OmniExecutorBase):
    """Generate narration/speech audio using Qwen3-Omni through vllm_omni.Omni."""

    name = EXECUTOR_NAME
    supported_task_types = frozenset({TaskType.OMNI_TEXT2GENERAL})
    _TASK_SPEC_TYPE = OmniText2GeneralSpecStrict

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return _HAS_OMNI and _HAS_VLLM

    def prepare(self) -> None:
        if not _HAS_OMNI:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2general executor."
            )
        if not _HAS_VLLM:
            raise ExecutionError(
                "vllm is not installed; "
                "omni_text2general requires SamplingParams from vllm."
            )

    def _run_inner(
        self,
        task: ExecutorTask,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniText2GeneralResult:
        assert isinstance(spec, OmniText2GeneralSpecStrict)
        texts = self._collect_text_inputs(spec, task.task_id)

        cfg = _narration_cfg(spec_dict)
        output_format = str(cfg.get("output_format") or "").strip().lower() or "wav"
        if output_format != "wav":
            raise ExecutionError(
                "omni_text2general currently supports output_format='wav' only."
            )
        sample_rate = to_int(cfg.get("sample_rate"), default=24000)
        output_modalities = _parse_modalities(cfg.get("modalities"))
        py_generator = to_bool(cfg.get("py_generator"), default=False)

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(texts)},
        ):
            self._ensure_omni(spec_dict)
        if self._omni is None:
            raise ExecutionError("Omni model failed to initialize.")

        prompts: list[OmniTextPrompt] = [
            {
                "prompt": self._build_prompt(text, spec_dict=spec_dict),
                "modalities": output_modalities,
            }
            for text in texts
        ]
        sampling_params = _build_sampling_params(cfg)

        audio_results: list[dict[str, Any]] = []
        text_results: dict[str, str] = {}
        stage_errors: list[str] = []
        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            generator: Iterable[OmniRequestOutput]
            try:
                if py_generator:
                    generator = self._omni.generate(
                        prompts, sampling_params, py_generator=True
                    )
                else:
                    generator = self._omni.generate(
                        prompts, sampling_params, py_generator=False
                    )
            except Exception as exc:
                raise ExecutionError(
                    f"omni_text2general generation failed to start: {exc}",
                    retryable=True,
                ) from exc

            for stage_output in generator:
                if stage_output.error:
                    stage_errors.append(stage_output.error)
                    continue
                final_type = stage_output.final_output_type
                if final_type == "text":
                    text_out = _extract_text_output(stage_output)
                    if text_out is not None:
                        text_results[stage_output.request_id] = text_out
                    continue
                if final_type == "audio":
                    audio_obj = extract_audio_from_mm(
                        extract_multimodal_output(stage_output)
                    )
                    if audio_obj is not None:
                        audio_results.append(
                            {
                                "request_id": stage_output.request_id,
                                "audio": audio_obj,
                            }
                        )

        if not audio_results:
            detail = (
                f" Engine reported: {'; '.join(stage_errors)}" if stage_errors else ""
            )
            raise ExecutionError(
                f"omni_text2general completed but returned no audio output.{detail}"
            )

        artifacts_dir = out_dir / "artifacts"
        items: list[OmniGeneralItem] = []
        multi = len(audio_results) > 1
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "item_count": len(audio_results)},
        ):
            for idx, entry in enumerate(audio_results):
                rid = str(entry.get("request_id") or f"req_{idx + 1}")
                audio_obj = entry.get("audio")
                save_path = self.resolve_save_path(
                    cfg,
                    out_dir,
                    index=idx,
                    ext=output_format,
                    multi=multi,
                    default_prefix="narration",
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(audio_obj, save_path, sample_rate=sample_rate)
                items.append(
                    OmniGeneralItem(
                        index=idx,
                        request_id=rid,
                        prompt=_prompt_for_request_id(rid, texts),
                        audio=ArtifactRef(
                            path=self.relative_to(save_path, artifacts_dir)
                        ),
                        text=text_results.get(rid) or None,
                    )
                )

        return OmniText2GeneralResult(
            model=self.model_name,
            items=items,
            audio=items[0].audio if items else None,
            sample_rate=sample_rate,
            storyboard=spec_dict.get("storyboard"),
        )

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        cfg = _narration_cfg(spec_dict)
        model_name = self.resolve_model_identifier(
            spec_dict,
            cfg,
            env_keys=("OMNI_NARRATION_MODEL",),
            default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        )
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

    # ── prompt ───────────────────────────────────────────────────────────

    def _build_prompt(self, text: str, spec_dict: dict[str, Any]) -> str:
        cfg = _narration_cfg(spec_dict)
        if to_bool(cfg.get("raw_prompt"), default=False):
            return text
        system_prompt = str(cfg.get("system_prompt") or _DEFAULT_SYSTEM_PROMPT).strip()
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


# ── narration-specific helpers ───────────────────────────────────────────────


def _narration_cfg(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return OmniExecutorBase.omni_cfg(spec_dict, "omni:narration", "omni_text2general")


def _prompt_index_from_request_id(request_id: str) -> int | None:
    """Recover the source prompt index from a vllm_omni request id.

    Request ids are built as ``f"{prompt_index}_{uuid4}"``, so the input prompt
    is correlated by id rather than by output position — one request can emit
    multiple audio chunks, breaking any positional pairing.
    """
    head = request_id.split("_", 1)[0]
    return int(head) if head.isdecimal() else None


def _prompt_for_request_id(request_id: str, texts: list[str]) -> str | None:
    """Return the input prompt that produced ``request_id``, or ``None``."""
    idx = _prompt_index_from_request_id(request_id)
    if idx is None or idx >= len(texts):
        return None
    return texts[idx]


def _parse_modalities(value: Any) -> list[str]:
    if value is None:
        return ["audio"]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or ["audio"]
    if isinstance(value, list):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return parts or ["audio"]
    return ["audio"]


def _build_sampling_params(cfg: dict[str, Any]) -> list[Any]:
    seed = to_int(cfg.get("seed"), default=42)
    talker_stop = to_int_list(cfg.get("talker_stop_token_ids"), default=[2150])
    thinker = SamplingParams(
        temperature=to_float(cfg.get("thinker_temperature"), default=0.9),
        top_p=to_float(cfg.get("thinker_top_p"), default=0.9),
        top_k=to_int(cfg.get("thinker_top_k"), default=-1),
        max_tokens=to_int(cfg.get("thinker_max_tokens"), default=1200),
        repetition_penalty=to_float(
            cfg.get("thinker_repetition_penalty"), default=1.05
        ),
        logit_bias={},
        seed=seed,
    )
    talker = SamplingParams(
        temperature=to_float(cfg.get("talker_temperature"), default=0.9),
        top_k=to_int(cfg.get("talker_top_k"), default=50),
        max_tokens=to_int(cfg.get("talker_max_tokens"), default=4096),
        seed=seed,
        detokenize=False,
        repetition_penalty=to_float(cfg.get("talker_repetition_penalty"), default=1.05),
        stop_token_ids=talker_stop,
    )
    code2wav = SamplingParams(
        temperature=to_float(cfg.get("code2wav_temperature"), default=0.0),
        top_p=to_float(cfg.get("code2wav_top_p"), default=1.0),
        top_k=to_int(cfg.get("code2wav_top_k"), default=-1),
        max_tokens=to_int(cfg.get("code2wav_max_tokens"), default=4096 * 16),
        seed=seed,
        detokenize=True,
        repetition_penalty=to_float(
            cfg.get("code2wav_repetition_penalty"), default=1.1
        ),
    )
    return [thinker, talker, code2wav]


def _extract_text_output(output: RequestOutput) -> str | None:
    if not (outputs := output.outputs):
        return None
    return text if (text := outputs[0].text.strip()) else None
