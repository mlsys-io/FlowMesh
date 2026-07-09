"""Tests for the Qwen3-TTS pure prompt/config helpers."""

from worker.executors.omni.qwen3_tts import (
    _cache_salt,
    _resolve_max_new_tokens,
    _resolve_task_type,
    is_qwen3_tts,
)


def test_is_qwen3_tts_matches_case_insensitively() -> None:
    assert is_qwen3_tts("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    assert is_qwen3_tts("  qwen3-tts-local  ")
    assert not is_qwen3_tts("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert not is_qwen3_tts("stabilityai/stable-audio-open-1.0")


def test_resolve_task_type_prefers_explicit() -> None:
    assert (
        _resolve_task_type(
            "Qwen/Qwen3-TTS-...-Base", cfg={"task_type": "VoiceDesign"}, data={}
        )
        == "VoiceDesign"
    )
    assert _resolve_task_type("model", cfg={}, data={"task_type": "  Base  "}) == "Base"


def test_resolve_task_type_falls_back_to_model_name() -> None:
    assert _resolve_task_type("Qwen/Qwen3-TTS-...-Base", cfg={}, data={}) == "Base"
    assert (
        _resolve_task_type("some-VoiceDesign-model", cfg={}, data={}) == "VoiceDesign"
    )
    assert (
        _resolve_task_type("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", cfg={}, data={})
        == "CustomVoice"
    )


def test_resolve_max_new_tokens_precedence_and_floor() -> None:
    assert _resolve_max_new_tokens(cfg={"max_new_tokens": 100}, data={}) == 100
    assert _resolve_max_new_tokens(cfg={"max_new_tokens": 5}, data={}) == 24  # floor
    assert _resolve_max_new_tokens(cfg={}, data={"max_new_tokens": 512}) == 512
    assert _resolve_max_new_tokens(cfg={"max_new_tokens": ""}, data={}) == 2048
    assert _resolve_max_new_tokens(cfg={}, data={}) == 2048


def test_cache_salt_is_deterministic_and_input_sensitive() -> None:
    a = {"text": ["hello"], "task_type": ["CustomVoice"], "speaker": ["vivian"]}
    b = {"text": ["hello"], "task_type": ["CustomVoice"], "speaker": ["vivian"]}
    c = {"text": ["goodbye"], "task_type": ["CustomVoice"], "speaker": ["vivian"]}

    salt_a = _cache_salt(a)
    assert len(salt_a) == 32
    assert all(ch in "0123456789abcdef" for ch in salt_a)
    assert salt_a == _cache_salt(b)
    assert salt_a != _cache_salt(c)


def test_cache_salt_tolerates_missing_keys() -> None:
    assert len(_cache_salt({})) == 32
