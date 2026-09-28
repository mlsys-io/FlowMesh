"""Tests for the workflow-level usage sum on GET /workflows/{id}."""

from pathlib import Path

from server.registries.workflow import Workflow, WorkflowStatus
from server.routers.v1.workflows import _task_usage, _workflow_usage
from shared.schemas.result import (
    AnyExecutorResult,
    APIResult,
    APIUsage,
    EchoResult,
    GenerationUsage,
    InferenceResult,
    ResultEnvelope,
    write_result,
)


def _write_result(results_dir: Path, task_id: str, result: AnyExecutorResult) -> None:
    write_result(
        results_dir,
        ResultEnvelope(task_id=task_id, result=result),
    )


def _workflow(completed: list[str]) -> Workflow:
    return Workflow(
        workflow_id="wfl-1",
        task_ids=completed,
        submitted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        status=WorkflowStatus.DONE,
        dispatched_tasks=[],
        completed_tasks=completed,
        failed_tasks=[],
        cancelled_tasks=[],
    )


def test_workflow_usage_sums_api_and_inference_tasks(
    tmp_path: Path,
) -> None:
    """Two finished tasks' usage sum into one workflow-level figure."""
    _write_result(
        tmp_path,
        "tsk-api",
        APIResult(
            ok=True,
            executor="api",
            method="POST",
            url="u",
            status_code=200,
            items=[],
            usage=APIUsage(
                prompt_tokens=30,
                completion_tokens=12,
                reasoning_tokens=2,
                calls=3,
                failures=0,
                retries=1,
                truncated_calls=1,
                wall_sec=2.5,
            ),
        ),
    )
    _write_result(
        tmp_path,
        "tsk-inf",
        InferenceResult(
            ok=True,
            model="m",
            items=[],
            usage=GenerationUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                num_requests=4,
                latency_sec=1.0,
            ),
        ),
    )

    usage = _workflow_usage(tmp_path, _workflow(["tsk-api", "tsk-inf"]))
    assert usage is not None
    assert usage.prompt_tokens == 130
    assert usage.completion_tokens == 62
    assert usage.reasoning_tokens == 2
    assert usage.calls == 7
    assert usage.retries == 1
    assert usage.truncated_calls == 1
    assert usage.wall_sec == 2.5


def test_workflow_usage_skips_tasks_without_usage(tmp_path: Path) -> None:
    """A task with no usage (echo) contributes nothing to the sum."""
    _write_result(
        tmp_path,
        "tsk-api",
        APIResult(
            ok=True,
            executor="api",
            method="POST",
            url="u",
            status_code=200,
            items=[],
            usage=APIUsage(
                prompt_tokens=5,
                completion_tokens=1,
                reasoning_tokens=0,
                calls=1,
                failures=0,
                retries=0,
                truncated_calls=0,
                wall_sec=0.1,
            ),
        ),
    )
    _write_result(
        tmp_path,
        "tsk-echo",
        EchoResult(ok=True, items=[], count=0),
    )

    usage = _workflow_usage(tmp_path, _workflow(["tsk-api", "tsk-echo"]))
    assert usage is not None
    assert usage.prompt_tokens == 5
    assert usage.calls == 1


def test_workflow_usage_none_when_no_task_reports_usage(tmp_path: Path) -> None:
    """A workflow whose finished tasks all lack usage has no usage object."""
    _write_result(
        tmp_path,
        "tsk-echo",
        EchoResult(ok=True, items=[], count=0),
    )
    assert _workflow_usage(tmp_path, _workflow(["tsk-echo"])) is None


def test_task_usage_returns_none_for_missing_result(tmp_path: Path) -> None:
    """A finished task with no stored result contributes nothing."""
    assert _task_usage(tmp_path, "tsk-missing") is None
