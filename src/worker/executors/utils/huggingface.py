"""Shared helpers for HuggingFace model / tokenizer loading."""

from typing import Any

import torch
from transformers import BitsAndBytesConfig


def pick_torch_dtype(training_cfg: dict[str, Any]) -> torch.dtype | None:
    if bool(training_cfg.get("bf16", False)) and torch.cuda.is_available():
        return torch.bfloat16
    if bool(training_cfg.get("fp16", False)) and torch.cuda.is_available():
        return torch.float16
    return None


def _pick_bnb_compute_dtype(torch_dtype: torch.dtype | None) -> torch.dtype:
    """Pick 4-bit compute dtype, falling back to fp16 on GPUs without bf16 support."""
    if torch_dtype is not None:
        return torch_dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_hf_load_kwargs(
    revision: str | None,
    trust_remote_code: bool,
    training_cfg: dict[str, Any],
    torch_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build ``(tok_kwargs, model_kwargs)`` for HuggingFace ``from_pretrained``."""
    tok_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if padding_side := training_cfg.get("padding_side"):
        tok_kwargs["padding_side"] = padding_side
    if trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True
    if revision:
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if bool(training_cfg.get("load_in_4bit", False)):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_pick_bnb_compute_dtype(torch_dtype),
            bnb_4bit_use_double_quant=bool(
                training_cfg.get("bnb_4bit_use_double_quant", True)
            ),
        )
    elif bool(training_cfg.get("load_in_8bit", False)):
        model_kwargs["load_in_8bit"] = True
    return tok_kwargs, model_kwargs
