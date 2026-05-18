"""Round-trip tests for the shared executor-result schema."""

import json
from typing import Any

from pydantic import Field

from shared.schemas.artifact import ArtifactContext, ArtifactRef
from shared.schemas.executor_result import BaseExecutorResult


class _SampleResult(BaseExecutorResult):
    ok: bool = True
    items: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None


def test_subclass_round_trip_through_base_preserves_extra_fields() -> None:
    """A subclass's executor-specific fields survive a JSON trip through the
    base class via ``extra='allow'``."""
    original = _SampleResult(
        ok=True,
        items=[{"output": "hello"}],
        usage={"latency_sec": 0.5},
        artifacts=ArtifactContext(base_dir="/tmp/t", base_url="http://h"),
    )
    payload = original.model_dump_json(serialize_as_any=True)

    base = BaseExecutorResult.model_validate_json(payload)
    redumped = json.loads(base.model_dump_json())

    assert redumped["items"] == [{"output": "hello"}]
    assert redumped["usage"] == {"latency_sec": 0.5}
    assert redumped["_artifacts"] == {"base_dir": "/tmp/t", "base_url": "http://h"}


def test_artifacts_alias_round_trips_both_directions() -> None:
    """The wire key is ``_artifacts`` (alias); field-name input is also accepted."""
    from_alias = BaseExecutorResult.model_validate({"_artifacts": {"base_dir": "/a"}})
    from_field = BaseExecutorResult.model_validate({"artifacts": {"base_dir": "/a"}})

    assert (
        from_alias.artifacts == from_field.artifacts == ArtifactContext(base_dir="/a")
    )

    dumped = from_alias.model_dump()
    assert "_artifacts" in dumped
    assert "artifacts" not in dumped


def test_recursive_children_round_trip() -> None:
    """Nested ``children`` deserialize as ``BaseExecutorResult`` and re-emit
    their extra fields when serialized with ``serialize_as_any=True``."""
    parent = _SampleResult(
        items=[{"output": "p"}],
        children={
            "c1": _SampleResult(items=[{"output": "c"}], usage={"total_tokens": 3}),
        },
    )
    payload = parent.model_dump_json(serialize_as_any=True)
    base = BaseExecutorResult.model_validate_json(payload)

    assert "c1" in base.children
    child = base.children["c1"]
    redumped = json.loads(child.model_dump_json())
    assert redumped["items"] == [{"output": "c"}]
    assert redumped["usage"] == {"total_tokens": 3}


def test_artifact_ref_is_a_typed_path_reference() -> None:
    ref = ArtifactRef(path="lora/final")
    assert ref.model_dump() == {"path": "lora/final"}


def test_envelope_round_trip_preserves_subclass_payload() -> None:
    """The production write→read path (``write_result_in_envelope`` →
    ``ResultEnvelope.model_validate``) round-trips subclass fields."""
    from shared.schemas.result import ResultEnvelope

    inner = _SampleResult(items=[{"output": "hello"}], usage={"total_tokens": 7})
    env = ResultEnvelope(task_id="tsk-x", result=inner)
    raw = env.model_dump_json(serialize_as_any=True)

    parsed = ResultEnvelope.model_validate_json(raw)
    dumped = parsed.result.model_dump()
    assert dumped["items"] == [{"output": "hello"}]
    assert dumped["usage"] == {"total_tokens": 7}
