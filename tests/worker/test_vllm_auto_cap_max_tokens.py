"""Tests for vLLM max_tokens auto-capping.

The pure clamp (``_auto_capped_max_tokens``) is builtins-only and runs without
vllm/torch. The wiring (``_auto_cap_sampling_params``) is exercised with a bare
instance + fakes, so it also needs no GPU deps.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from worker.executors.vllm_executor import (
    _AUTO_CAP_MIN_OUTPUT,
    VLLMExecutor,
    _auto_capped_max_tokens,
)


# ── pure clamp ───────────────────────────────────────────────────────────
def test_clamp_fits_output_to_window() -> None:
    # the field bug: 65536 requested on a 40960 window with an 8193-tok prompt
    assert _auto_capped_max_tokens(65536, 8193, 40960) == 40960 - 8193 - 16


def test_clamp_untouched_when_window_unknown() -> None:
    assert _auto_capped_max_tokens(65536, 8193, None) == 65536
    assert _auto_capped_max_tokens(65536, 8193, 0) == 65536


def test_clamp_untouched_when_request_already_fits() -> None:
    assert _auto_capped_max_tokens(512, 8193, 40960) == 512


def test_clamp_floors_when_prompt_at_or_over_window() -> None:
    # prompt at/over the window → floor; vLLM then raises the honest
    # "maximum context length" error rather than us hiding it.
    assert _auto_capped_max_tokens(65536, 40960, 40960) == _AUTO_CAP_MIN_OUTPUT
    assert _auto_capped_max_tokens(65536, 50000, 40960) == _AUTO_CAP_MIN_OUTPUT


def test_clamp_never_exceeds_requested_even_below_floor() -> None:
    # a below-floor request stays capped at the request, never raised to floor
    assert _auto_capped_max_tokens(8, 40950, 40960) == 8


# ── wiring ───────────────────────────────────────────────────────────────
def _executor_with(
    window: int | None,
    prompts: list[Any],
    tok_len: int | Callable[[str], int],
) -> VLLMExecutor:
    ex = object.__new__(VLLMExecutor)  # skip __init__ (no GPU deps)
    ex._batched_inputs = prompts
    ex._llm = SimpleNamespace(  # type: ignore[assignment]
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=window))
    )

    def _encode(s: str) -> list[int]:
        # tok_len may be a constant or a per-prompt callable.
        return [0] * (tok_len(s) if callable(tok_len) else tok_len)

    ex._get_tokenizer = lambda: SimpleNamespace(  # type: ignore[method-assign]
        encode=_encode
    )
    return ex


def _sp(max_tokens: int) -> MagicMock:
    sp = MagicMock()
    sp.max_tokens = max_tokens
    # A fresh clone per call, mirroring SamplingParams.clone()'s deep copy —
    # a shared mock would let one clamp overwrite another.
    sp.clone.side_effect = lambda: SimpleNamespace(max_tokens=None)
    return sp


def test_wiring_returns_shared_object_when_window_unknown() -> None:
    ex = _executor_with(None, ["hello"], 5)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_returns_shared_object_when_nothing_clamped() -> None:
    ex = _executor_with(40960, ["a", "b"], 10)  # 10 + 512 << 40960
    sp = _sp(512)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_builds_per_prompt_list_and_clamps() -> None:
    ex = _executor_with(40960, ["big-prompt"], 8193)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].max_tokens == 40960 - 8193 - 16
    assert capped == {0: {"max_tokens": 40960 - 8193 - 16, "requested": 65536}}


def test_wiring_leaves_multimodal_prompts_at_requested() -> None:
    # non-str (TextPrompt-shaped) entries keep the requested budget
    ex = _executor_with(40960, [{"prompt": "x", "multi_modal_data": {}}], 8193)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


def test_wiring_clamps_only_the_oversized_prompt_in_a_mixed_batch() -> None:
    # requested=35000 fits after the 10-tok prompt but not the 8193-tok one, so
    # "big" clamps while "small" fits and shares the original object.
    ex = _executor_with(40960, ["big", "small"], lambda s: 8193 if s == "big" else 10)
    sp = _sp(35000)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert isinstance(out, list) and len(out) == 2
    assert out[0].max_tokens == 40960 - 8193 - 16
    assert out[1] is sp
    assert capped == {0: {"max_tokens": 40960 - 8193 - 16, "requested": 35000}}


def test_wiring_tokenizer_failure_skips_clamp() -> None:
    def _raise(_s: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    ex = _executor_with(40960, ["big-prompt"], _raise)
    sp = _sp(65536)
    out, capped = ex._auto_cap_sampling_params(sp)
    assert out is sp
    assert capped == {}


# ── grouped-output remap ─────────────────────────────────────────────────
def _grouped_executor() -> VLLMExecutor:
    return object.__new__(VLLMExecutor)


def _item(output: str, *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"output": output, "finish_reason": "stop"}
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def test_grouped_remap_aggregates_auto_cap_per_member() -> None:
    ex = _grouped_executor()
    items = [
        _item("a", diagnostics={"auto_cap": {"max_tokens": 32751, "requested": 65536}}),
        _item("b"),
    ]
    out = ex._remap_grouped_outputs(
        task_id="tsk-1",
        items=items,
        group_sizes=[2],
        base_prompts=["grouped"],
        base_metadata=[{}],
    )
    assert out[0]["diagnostics"] == {
        "auto_cap": {"max_tokens": [32751, None], "requested": 65536}
    }


def test_grouped_remap_omits_diagnostics_when_no_member_capped() -> None:
    ex = _grouped_executor()
    out = ex._remap_grouped_outputs(
        task_id="tsk-1",
        items=[_item("a"), _item("b")],
        group_sizes=[2],
        base_prompts=["grouped"],
        base_metadata=[{}],
    )
    assert "diagnostics" not in out[0]
