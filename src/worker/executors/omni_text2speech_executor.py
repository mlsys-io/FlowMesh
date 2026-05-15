"""Omni executor for text-to-speech via vllm_omni."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from vllm_omni.entrypoints.omni import Omni

    _HAS_OMNI = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.entrypoints.omni import Omni
    else:
        Omni = None
    _HAS_OMNI = False

from shared.schemas.governance import SpanType
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2SpeechSpecStrict
from shared.utils.parsing import as_list, to_int

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import (
    OmniExecutorBase,
    extract_audio_from_mm,
    extract_multimodal_output,
    save_audio,
)
from .utils.checkpoints import artifact_ref

logger = logging.getLogger(__name__)


class OmniText2SpeechExecutor(OmniExecutorBase):
    """Generate speech audio using vllm_omni.Omni."""

    name = "omni_text2speech"
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
    ) -> dict[str, Any]:
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
                        "audio": artifact_ref(
                            self.relative_to(save_path, artifacts_dir)
                        ),
                    }
                )

        first = items[0]["audio"] if items else {}
        result: dict[str, Any] = {
            "ok": True,
            "executor": self.name,
            "mode": "tts",
            "model": self._model_name,
            "audio": first,
            "items": items,
            "sample_rate": sample_rate,
        }
        storyboard = spec_dict.get("storyboard")
        if isinstance(storyboard, dict):
            result["storyboard"] = dict(storyboard)
        return result

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        cfg = self.omni_cfg(spec_dict, "omni:tts", "omni_text2speech")
        model_name = self.resolve_model_identifier(
            spec_dict,
            cfg,
            env_keys=("OMNI_TTS_MODEL",),
            default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
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

    # ── generation ───────────────────────────────────────────────────────

    def _generate_single(self, text: str, spec_dict: dict[str, Any]) -> Any:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        prompt = self._build_tts_prompt(text, spec_dict=spec_dict)
        outputs = self._omni.generate(prompt, use_tqdm=False)
        audio = _extract_first_audio(outputs)
        if audio is None:
            raise ExecutionError("Omni TTS generation returned no audio.")
        return audio

    def _build_tts_prompt(self, text: str, spec_dict: dict[str, Any]) -> Any:
        model_name = str(self._model_name or "")
        if "qwen3-tts" not in model_name.strip().lower():
            return text
        cfg = self.omni_cfg(spec_dict, "omni:tts", "omni_text2speech")
        data = spec_dict.get("data") if isinstance(spec_dict.get("data"), dict) else {}
        if not isinstance(data, dict):
            data = {}

        task_type = _resolve_qwen3_tts_task_type(model_name, cfg=cfg, data=data)
        additional: dict[str, Any] = {
            "task_type": [task_type],
            "text": [text],
            "max_new_tokens": [
                _resolve_max_new_tokens(
                    cfg=cfg, data=data, text=text, task_type=task_type
                )
            ],
        }
        language = str(
            cfg.get("language")
            or data.get("language")
            or os.getenv("OMNI_TTS_LANGUAGE")
            or "Auto"
        ).strip()
        if language:
            additional["language"] = [language]
        instruct = str(
            cfg.get("instruct")
            or cfg.get("instructions")
            or data.get("instruct")
            or data.get("instructions")
            or ""
        ).strip()
        if instruct:
            additional["instruct"] = [instruct]
        if task_type == "CustomVoice":
            speaker = str(
                cfg.get("speaker")
                or cfg.get("voice")
                or data.get("speaker")
                or data.get("voice")
                or os.getenv("OMNI_TTS_VOICE")
                or "Vivian"
            ).strip()
            additional["speaker"] = [speaker]

        return {"prompt_token_ids": [0] * 2048, "additional_information": additional}


# ── TTS-specific helpers ─────────────────────────────────────────────────────


def _extract_first_audio(outputs: Any) -> Any:
    for stage in as_list(outputs):
        ros = as_list(getattr(stage, "request_output", None))
        if ros:
            for ro in ros:
                mm = extract_multimodal_output(ro)
                audio = extract_audio_from_mm(mm)
                if audio is not None:
                    return audio
            continue
        mm = extract_multimodal_output(stage)
        audio = extract_audio_from_mm(mm)
        if audio is not None:
            return audio
    return None


def _resolve_qwen3_tts_task_type(
    model_name: str, cfg: dict[str, Any], data: dict[str, Any]
) -> str:
    explicit = cfg.get("task_type") or data.get("task_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    lower = model_name.strip().lower()
    if "-base" in lower:
        return "Base"
    if "voicedesign" in lower:
        return "VoiceDesign"
    return "CustomVoice"


def _resolve_max_new_tokens(
    cfg: dict[str, Any], data: dict[str, Any], text: str, task_type: str
) -> int:
    raw = cfg.get("max_new_tokens")
    if raw in (None, ""):
        raw = data.get("max_new_tokens")
    if raw not in (None, ""):
        return max(24, to_int(raw, default=512))
    if task_type in {"CustomVoice", "VoiceDesign"}:
        return int(max(512, max(192, min(768, len(text.strip()) * 6))))
    return 2048
