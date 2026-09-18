"""Tests for task merge key computation and spec sanitization."""

from server.task.runtime import _compute_merge_key, _sanitize_merge_spec
from shared.tasks import TaskEnvelopeTemplate


class TestSanitizeMergeSpec:
    def test_strips_system_prompt(self) -> None:
        spec = {
            "taskType": "inference",
            "inference": {"system_prompt": "secret", "temperature": 0.7},
            "data": {"messages": [{"role": "user", "content": "hi"}]},
        }
        result = _sanitize_merge_spec(spec)
        assert "system_prompt" not in result.get("inference", {})
        assert "data" not in result  # data is stripped too

    def test_preserves_other_fields(self) -> None:
        spec = {
            "taskType": "inference",
            "inference": {"temperature": 0.7, "max_tokens": 100},
            "model": {"source": {"identifier": "llama"}},
        }
        result = _sanitize_merge_spec(spec)
        assert result["inference"]["temperature"] == 0.7
        assert result["model"]["source"]["identifier"] == "llama"

    def test_no_inference_key(self) -> None:
        spec = {"taskType": "echo", "data": {"items": ["x"]}}
        result = _sanitize_merge_spec(spec)
        assert "data" not in result
        assert result["taskType"] == "echo"


class TestComputeMergeKey:
    def _make_task(
        self, task_type: str = "inference", **spec_kw
    ) -> TaskEnvelopeTemplate:
        spec_data = {"taskType": task_type, **spec_kw}
        return TaskEnvelopeTemplate.model_validate(
            {"apiVersion": "flowmesh/v1", "kind": "Task", "spec": spec_data}
        )

    def test_deterministic(self) -> None:
        t = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            inference={"temperature": 0.7},
        )
        k1 = _compute_merge_key(t)
        k2 = _compute_merge_key(t)
        assert k1 is not None
        assert k1 == k2

    def test_different_specs_different_keys(self) -> None:
        t1 = self._make_task("inference", model={"source": {"identifier": "llama"}})
        t2 = self._make_task("inference", model={"source": {"identifier": "gpt-4"}})
        k1 = _compute_merge_key(t1)
        k2 = _compute_merge_key(t2)
        assert k1 != k2

    def test_non_mergeable_type_returns_none(self) -> None:
        t = self._make_task("echo")
        assert _compute_merge_key(t) is None

    def test_ignores_data_field(self) -> None:
        """Two tasks with same model but different prompts should merge."""
        t1 = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            data={"messages": [{"role": "user", "content": "hello"}]},
        )
        t2 = self._make_task(
            "inference",
            model={"source": {"identifier": "llama"}},
            data={"messages": [{"role": "user", "content": "goodbye"}]},
        )
        k1 = _compute_merge_key(t1)
        k2 = _compute_merge_key(t2)
        assert k1 is not None
        assert k1 == k2
