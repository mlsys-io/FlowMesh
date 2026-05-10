"""Unit tests for VLLMExecutor's model.source.revision fallback."""

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")

from worker.executors.vllm_executor import _resolve_engine_revision


def test_vllm_scoped_revision_wins() -> None:
    assert _resolve_engine_revision("abc123", "main") == "abc123"


def test_falls_back_to_source_revision() -> None:
    assert _resolve_engine_revision(None, "main") == "main"


def test_returns_none_when_neither_set() -> None:
    assert _resolve_engine_revision(None, None) is None


def test_empty_source_revision_treated_as_unset() -> None:
    assert _resolve_engine_revision(None, "") is None
