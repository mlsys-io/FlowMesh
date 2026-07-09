"""Qwen3-TTS prompt and sampling-parameter construction for the TTS executor.

Mirrors the talker-prompt contract vllm_omni drives Qwen3-TTS with: a placeholder
token-id prompt sized to the model-side ``inputs_embeds`` estimate, the real
conditioning carried in ``additional_information``, a ``cache_salt`` so the
placeholder ids don't collide in the prefix cache, and a ``tts_local_seed`` folded
into the stage-0 sampling params.
"""

import copy
import hashlib
import logging
import os
from functools import lru_cache
from typing import Any

from shared.utils.parsing import to_int

from ..omni_executor_base import Omni

logger = logging.getLogger(__name__)


def is_qwen3_tts(model_name: str) -> bool:
    return "qwen3-tts" in model_name.strip().lower()


def build_prompt(
    *, model_name: str, text: str, cfg: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    from vllm.inputs import tokens_input

    task_type = _resolve_task_type(model_name, cfg=cfg, data=data)
    additional: dict[str, Any] = {
        "task_type": [task_type],
        "text": [text],
        "max_new_tokens": [_resolve_max_new_tokens(cfg=cfg, data=data)],
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
        additional["speaker"] = [speaker.lower()]
        additional["voice_created_at"] = [0]

    prompt_len = _estimate_prompt_len(
        model_name=model_name, additional=additional, task_type=task_type
    )
    prompt = dict(tokens_input(prompt_token_ids=[1] * prompt_len))
    prompt["additional_information"] = additional
    prompt["cache_salt"] = _cache_salt(additional)
    prompt["modalities"] = ["audio"]
    return prompt


def build_sampling_params(omni: Omni) -> list[Any]:
    from vllm_omni.entrypoints.utils import coerce_param_message_types

    params = copy.deepcopy(list(omni.default_sampling_params_list))
    params = coerce_param_message_types(params, is_streaming=False)
    if params:
        stage0_params = params[0]
        default_seed = stage0_params.seed
        if default_seed is not None:
            if stage0_params.extra_args is None:
                stage0_params.extra_args = {}
            stage0_params.extra_args.setdefault("tts_local_seed", int(default_seed))
    return list(params)


def _cache_salt(additional: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in (
        "text",
        "task_type",
        "language",
        "speaker",
        "voice_created_at",
        "ref_text",
        "instruct",
        "x_vector_only_mode",
    ):
        h.update(b"\x00")
        value = additional.get(key)
        if value is not None:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()[:32]


@lru_cache(maxsize=4)
def _prompt_assets(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )

    config = Qwen3TTSConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    return config, tokenizer


def _estimate_prompt_len(
    *, model_name: str, additional: dict[str, Any], task_type: str
) -> int:
    try:
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        config, tokenizer = _prompt_assets(model_name)
        talker_config = config.talker_config
        return (
            Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=additional,
                task_type=task_type,
                tokenize_prompt=lambda text: tokenizer(text, padding=False)[
                    "input_ids"
                ],
                codec_language_id=getattr(talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
            )
        )
    except Exception as exc:
        logger.warning(
            "Failed to estimate Qwen3-TTS prompt length; using fallback 2048: %s",
            exc,
        )
        return 2048


def _resolve_task_type(
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


def _resolve_max_new_tokens(cfg: dict[str, Any], data: dict[str, Any]) -> int:
    raw = cfg.get("max_new_tokens")
    if raw in (None, ""):
        raw = data.get("max_new_tokens")
    if raw not in (None, ""):
        return max(24, to_int(raw, default=512))
    return 2048
