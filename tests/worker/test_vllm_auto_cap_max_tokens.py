"""Tests for vLLM max_tokens auto-capping.

The pure clamp (``_auto_capped_max_tokens``) is builtins-only and runs without
vllm/torch. The wiring (``_auto_cap_sampling_params``) is exercised with a bare
instance + fakes, so it also needs no GPU deps.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from worker.executors.vllm_executor import (
    VLLMExecutor,
    _AUTO_CAP_MIN_OUTPUT,
    _auto_capped_max_tokens,
)


# ── pure clamp ───────────────────────────────────────────────────────────
def test_clamp_fits_output_to_window():
    # the field bug: 65536 requested on a 40960 window with an 8193-tok prompt
    assert _auto_capped_max_tokens(65536, 8193, 40960) == 40960 - 8193 - 16


def test_clamp_untouched_when_window_unknown():
    assert _auto_capped_max_tokens(65536, 8193, None) == 65536
    assert _auto_capped_max_tokens(65536, 8193, 0) == 65536


def test_clamp_untouched_when_request_already_fits():
    assert _auto_capped_max_tokens(512, 8193, 40960) == 512


def test_clamp_floors_when_prompt_at_or_over_window():
    # prompt at/over the window → floor; vLLM then raises the honest
    # "maximum context length" error rather than us hiding it.
    assert _auto_capped_max_tokens(65536, 40960, 40960) == _AUTO_CAP_MIN_OUTPUT
    assert _auto_capped_max_tokens(65536, 50000, 40960) == _AUTO_CAP_MIN_OUTPUT


# ── wiring ───────────────────────────────────────────────────────────────
def _executor_with(window, prompts, tok_len):
    ex = object.__new__(VLLMExecutor)  # skip __init__ (no GPU deps)
    ex._batched_inputs = prompts
    ex._llm = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=window))
    )
    ex._get_tokenizer = lambda: SimpleNamespace(  # type: ignore[method-assign]
        encode=lambda s: [0] * tok_len
    )
    return ex


def _sp(max_tokens):
    sp = MagicMock()
    sp.max_tokens = max_tokens
    clone = MagicMock()
    clone.max_tokens = max_tokens
    sp.clone.return_value = clone
    return sp


def test_wiring_returns_shared_object_when_window_unknown():
    ex = _executor_with(None, ["hello"], 5)
    sp = _sp(65536)
    assert ex._auto_cap_sampling_params(sp) is sp


def test_wiring_returns_shared_object_when_nothing_clamped():
    ex = _executor_with(40960, ["a", "b"], 10)  # 10 + 512 << 40960
    sp = _sp(512)
    assert ex._auto_cap_sampling_params(sp) is sp


def test_wiring_builds_per_prompt_list_and_clamps():
    ex = _executor_with(40960, ["big-prompt"], 8193)
    sp = _sp(65536)
    out = ex._auto_cap_sampling_params(sp)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].max_tokens == 40960 - 8193 - 16


def test_wiring_leaves_multimodal_prompts_at_requested():
    # non-str (TextPrompt-shaped) entries keep the requested budget
    ex = _executor_with(40960, [{"prompt": "x", "multi_modal_data": {}}], 8193)
    sp = _sp(65536)
    assert ex._auto_cap_sampling_params(sp) is sp
