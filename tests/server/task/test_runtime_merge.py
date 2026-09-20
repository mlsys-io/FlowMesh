"""Task merging: merge-key computation and spec sanitization (pure functions),
plus runtime merge planning and atomic merged-child failure.

Credential retention is gated at dispatch, not at merge time: ``plan_merge``
stays redaction-agnostic and a redacted child is failed individually during
merge resolution via ``fail_merged_children``, which unlinks the child and fails
it with its dependent cascade, persisting the failed records before the parent's
unlink (children-first) — so a well-shaped task is never sunk by a bad sibling
and a crash cannot leave the parent unlinked from a child that is not yet failed.
Merge keys are spec-content-based, so tasks from different workflows on the same
model coalesce; because ``commit_transition`` is per-workflow, each merged
sibling's dispatch membership must commit under its own workflow.
"""

from server.task.models import TaskStatus
from server.task.runtime import _compute_merge_key, _sanitize_merge_spec
from shared.tasks import TaskEnvelopeTemplate

from .merge_harness import build_runtime, register

_WORKER = "wkr-1"

_MERGE_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: merge-batch
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
      - name: b
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
      - name: c
        dependsOn: [b]
        spec:
          taskType: echo
"""

_SINGLE_INFER_PAYLOAD = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: single-infer
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: inference
          model:
            source:
              identifier: llama
"""


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


def test_clean_siblings_merge() -> None:
    runtime, _ = build_runtime()
    _, nodes = register(runtime, _MERGE_PAYLOAD)
    assert runtime.plan_merge(nodes["a"], 2, _WORKER) == [nodes["b"]]


def test_fail_merged_children_fails_child_and_cascades_atomically() -> None:
    runtime, registry = build_runtime()
    _, nodes = register(runtime, _MERGE_PAYLOAD)
    parent, child, dep = nodes["a"], nodes["b"], nodes["c"]
    runtime.plan_merge(parent, 2, _WORKER)
    assert runtime._tasks[parent].merged_children == [child]
    registry.calls.clear()

    failed, impacted = runtime.fail_merged_children(
        parent, [child], "credential_not_retained"
    )

    assert failed == [child]
    assert impacted == [(dep, f"Dependency {child} failed")]
    # Child is unlinked from the parent and failed; its dependent cascades.
    assert runtime._tasks[parent].merged_children is None
    assert runtime._merge_parent_map.get(child) is None
    assert runtime._tasks[child].status == TaskStatus.FAILED
    assert runtime._tasks[child].merged_parent_id is None
    assert runtime._tasks[dep].status == TaskStatus.FAILED
    # Child and dependent (same workflow) fail together in one transaction.
    failed_idx = next(
        i for i, c in enumerate(registry.calls) if set(c["failed"]) == {child, dep}
    )
    assert set(registry.calls[failed_idx]["record_ids"]) >= {child, dep}
    # Crash-safe ordering: the parent's unlink commits after its children are failed.
    parent_idx = next(
        i for i, c in enumerate(registry.calls) if parent in c["record_ids"]
    )
    assert failed_idx < parent_idx


def test_plan_merge_persists_cross_workflow_sibling_under_its_own_workflow() -> None:
    runtime, registry = build_runtime()
    wf1, n1 = register(runtime, _SINGLE_INFER_PAYLOAD)
    wf2, n2 = register(runtime, _SINGLE_INFER_PAYLOAD)
    t1, t2 = n1["a"], n2["a"]
    assert wf1 != wf2
    registry.calls.clear()

    merged = runtime.plan_merge(t1, 4, _WORKER)

    # The sibling from the other workflow is batched.
    assert merged == [t2]
    # Its dispatch membership commits under wf2, never under the parent's wf1.
    assert any(
        c["workflow_id"] == wf2 and t2 in c["dispatched"] for c in registry.calls
    )
    assert all(
        t2 not in c["dispatched"] for c in registry.calls if c["workflow_id"] == wf1
    )
    # The parent's record is still persisted, under its own workflow.
    assert any(
        c["workflow_id"] == wf1 and t1 in c["record_ids"] for c in registry.calls
    )
