"""Tests for Omni multimodal output extraction helpers."""

from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.omni_executor_base import (
    OmniRequestOutput,
    extract_audio_from_mm,
    extract_multimodal_output,
)
from worker.executors.omni_text2audio_executor import _extract_audio_waveforms
from worker.executors.omni_text2general_executor import (
    _prompt_for_request_id,
    _prompt_index_from_request_id,
)


class MappingPayload(Mapping[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class FakeCompletion:
    def __init__(
        self, *, multimodal_output: Any = None, text: str | None = None
    ) -> None:
        self.multimodal_output = multimodal_output
        self.text = text


class FakeOmniRequestOutput:
    """Mirror of OmniRequestOutput's output-access contract.

    The completion outputs hang off the request output itself, and
    ``multimodal_output`` surfaces the payload attached to the first completion
    before falling back to the request-level payload.
    """

    def __init__(
        self,
        *,
        request_id: str = "req",
        outputs: list[FakeCompletion] | None = None,
        multimodal_output: Any = None,
    ) -> None:
        self.request_id = request_id
        self.outputs = outputs or []
        self._multimodal_output = multimodal_output

    @property
    def multimodal_output(self) -> Any:
        for completion in self.outputs:
            if mm := completion.multimodal_output:
                return mm
        return self._multimodal_output

    def as_omni_request_output(self) -> OmniRequestOutput:
        return cast(OmniRequestOutput, self)


def test_extract_audio_from_mm_accepts_audio_payload() -> None:
    payload = {"audio": [0.1, -0.1], "sample_rate": 24000}

    assert extract_audio_from_mm(payload) == {
        "audio": [0.1, -0.1],
        "sample_rate": 24000,
    }


def test_extract_audio_from_mm_accepts_qwen3_tts_model_outputs() -> None:
    payload = {"model_outputs": [[0.1, -0.1]], "sr": [24000]}

    assert extract_audio_from_mm(payload) == {
        "audio": [[0.1, -0.1]],
        "sample_rate": 24000,
    }


def test_extract_audio_from_mm_accepts_mapping_payload() -> None:
    payload = MappingPayload({"model_outputs": [[0.1, -0.1]], "sr": [24000]})

    assert extract_audio_from_mm(payload) == {
        "audio": [[0.1, -0.1]],
        "sample_rate": 24000,
    }


def test_extract_multimodal_output_returns_completion_mapping_payload() -> None:
    payload = MappingPayload({"model_outputs": [[0.1, -0.1]], "sr": [24000]})
    output = FakeOmniRequestOutput(
        outputs=[FakeCompletion(multimodal_output=payload)]
    ).as_omni_request_output()

    assert extract_multimodal_output(output) is payload


def test_extract_multimodal_output_rejects_non_mapping_payload() -> None:
    output = FakeOmniRequestOutput(
        outputs=[FakeCompletion(multimodal_output=[0.1, -0.1])]
    ).as_omni_request_output()

    assert extract_multimodal_output(output) is None


def test_text2audio_extracts_mapping_payload_audio() -> None:
    payload = MappingPayload({"audio": [0.1, -0.1]})
    output = FakeOmniRequestOutput(
        request_id="req-1",
        outputs=[FakeCompletion(multimodal_output=payload)],
    ).as_omni_request_output()

    extracted = _extract_audio_waveforms([output])

    assert len(extracted) == 1
    assert extracted[0]["request_id"] == "req-1"
    assert extracted[0]["waveform"].tolist() == pytest.approx([0.1, -0.1])


def test_text2audio_extracts_model_outputs_fallback() -> None:
    payload = MappingPayload({"model_outputs": [0.1, -0.1]})
    output = FakeOmniRequestOutput(
        request_id="req-2",
        outputs=[FakeCompletion(multimodal_output=payload)],
    ).as_omni_request_output()

    extracted = _extract_audio_waveforms([output])

    assert len(extracted) == 1
    assert extracted[0]["request_id"] == "req-2"
    assert extracted[0]["waveform"].tolist() == pytest.approx([0.1, -0.1])


def test_text2general_extracts_mapping_payload_model_outputs() -> None:
    payload = MappingPayload({"model_outputs": [[0.1, -0.1]], "sr": [24000]})
    output = FakeOmniRequestOutput(
        outputs=[FakeCompletion(multimodal_output=payload)]
    ).as_omni_request_output()

    assert extract_audio_from_mm(extract_multimodal_output(output)) == {
        "audio": [[0.1, -0.1]],
        "sample_rate": 24000,
    }


def test_prompt_index_parses_leading_segment() -> None:
    assert _prompt_index_from_request_id("0_f47ac10b-58cc") == 0
    assert _prompt_index_from_request_id("3_deadbeef") == 3


def test_prompt_index_is_none_for_unindexed_request_id() -> None:
    assert _prompt_index_from_request_id("req_1") is None
    assert _prompt_index_from_request_id("f47ac10b") is None
    assert _prompt_index_from_request_id("") is None


def test_prompt_for_request_id_correlates_chunks_to_source_prompt() -> None:
    texts = ["first prompt", "second prompt"]
    request_ids = ["0_uuidA", "0_uuidA", "1_uuidB"]

    prompts = [_prompt_for_request_id(rid, texts) for rid in request_ids]

    assert prompts == ["first prompt", "first prompt", "second prompt"]


def test_prompt_for_request_id_raises_when_uncorrelated() -> None:
    texts = ["only prompt"]

    with pytest.raises(ExecutionError):
        _prompt_for_request_id("5_uuid", texts)  # index out of range
    with pytest.raises(ExecutionError):
        _prompt_for_request_id("req_1", texts)  # no leading index
