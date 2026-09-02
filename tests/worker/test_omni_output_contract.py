"""Contract checks against the installed vllm-omni.

The extraction helpers are otherwise exercised only through hand-rolled fakes,
which cannot notice an upstream API change. These construct real
``OmniRequestOutput`` values and assert the shape the omni executors read:
a stage output *is* a vLLM ``RequestOutput``, carrying its own completion
outputs and request id.
"""

from typing import Any

import pytest

pytest.importorskip("vllm_omni", reason="vllm-omni not installed")

from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm_omni.outputs import OmniRequestOutput  # noqa: E402

from worker.executors.omni_executor_base import (  # noqa: E402
    extract_audio_from_mm,
    extract_multimodal_output,
)


def _completion(*, text: str = "", multimodal_output: Any = None) -> CompletionOutput:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1],
        cumulative_logprob=None,
        logprobs=None,
    )
    if multimodal_output is not None:
        completion.multimodal_output = multimodal_output  # type: ignore[attr-defined]
    return completion


def _stage_output(
    completions: list[CompletionOutput],
    *,
    request_id: str = "req-1",
    final_output_type: str = "text",
) -> OmniRequestOutput:
    source = RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=completions,
        finished=True,
    )
    return OmniRequestOutput.from_stage_output(
        source, final_output_type=final_output_type
    )


def test_stage_output_is_a_request_output() -> None:
    output = _stage_output([_completion(text="hello")])

    assert isinstance(output, RequestOutput)
    assert output.request_id == "req-1"
    assert output.final_output_type == "text"
    assert [completion.text for completion in output.outputs] == ["hello"]


def test_stage_output_surfaces_completion_multimodal_payload() -> None:
    payload = {"audio": [0.1, -0.1], "sample_rate": 24000}
    output = _stage_output(
        [_completion(multimodal_output=payload)], final_output_type="audio"
    )

    assert extract_multimodal_output(output) == payload
    assert extract_audio_from_mm(extract_multimodal_output(output)) == {
        "audio": [0.1, -0.1],
        "sample_rate": 24000,
    }


def test_error_output_carries_the_engine_message() -> None:
    output = OmniRequestOutput.from_error("req-2", "stage 1 exploded")

    assert output.error == "stage 1 exploded"
    assert output.outputs == []
    assert output.finished
