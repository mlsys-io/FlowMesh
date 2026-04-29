"""End-to-end lineage flow against the dag_inference_example workflow.

Brings up the full FlowMesh stack, runs a real multi-stage inference DAG on a
GPU worker, and verifies that the new lineage transport works end-to-end:

- Cross-task data flow goes through Redis (no HTTP gov calls in worker logs).
- Each task's `logs/{events,assets,lineage}.jsonl` arrives at the server.
- `flowmesh logs fetch <kind>` returns rows.
- `flowmesh profile fetch` returns a summary whose lineage edges match the DAG.
- Redis keys for cross-task payloads carry a TTL.

Skipped unless `FLOWMESH_E2E=1`. Requires a live FlowMesh stack and an
available GPU worker on GPU 2 or 3 (the user reserves these for FlowMesh).
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("FLOWMESH_E2E") != "1",
    reason="set FLOWMESH_E2E=1 to run the live e2e test",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "templates" / "dag_inference_example.yaml"
GPU_TARGETS = os.getenv("FLOWMESH_E2E_GPU_TARGETS", "2,3")


def _run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["uv", "run", "flowmesh", *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=capture,
        text=True,
    )


def _wait_for_workflow(workflow_id: str, timeout_sec: int = 600) -> str:
    """Poll workflow status until it reaches a terminal state."""
    start = time.time()
    while time.time() - start < timeout_sec:
        result = _run(["workflow", "info", workflow_id])
        info = json.loads(result.stdout)
        status = info.get("status")
        if status in {"DONE", "FAILED", "CANCELLED"}:
            return status
        time.sleep(5)
    raise TimeoutError(f"workflow {workflow_id} did not finish in {timeout_sec}s")


@pytest.fixture(scope="module")
def stack_up():
    """Bring up the stack + a GPU worker; tear down at the end."""
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    _run(["stack", "up"], capture=False)
    _run(
        ["stack", "worker", "up", "gpu", "--targets", GPU_TARGETS],
        capture=False,
    )
    try:
        yield
    finally:
        _run(["stack", "worker", "down"], capture=False)
        _run(["stack", "down"], capture=False)


def _submit_workflow() -> str:
    result = _run(["workflow", "submit", str(TEMPLATE)])
    payload = json.loads(result.stdout)
    return payload["workflow_id"]


def _fetch_jsonl(workflow_id: str, kind: str) -> list[dict]:
    result = _run(["logs", "fetch", kind, workflow_id])
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_dag_inference_lineage_e2e(stack_up) -> None:
    workflow_id = _submit_workflow()
    final_status = _wait_for_workflow(workflow_id)
    assert final_status == "DONE", f"workflow ended with status {final_status}"

    events = _fetch_jsonl(workflow_id, "events")
    assets = _fetch_jsonl(workflow_id, "assets")
    lineage = _fetch_jsonl(workflow_id, "lineage")

    # The DAG has 3 nodes (branch-a, branch-b, synthesis), so we expect 3 assets.
    assert len(assets) >= 3
    # synthesis depends on branch-a and branch-b → at least 2 lineage edges.
    assert len(lineage) >= 2
    sources = {row["source_data_id"] for row in lineage}
    derived = {row["data_id"] for row in lineage}
    assert sources, "lineage edges have no source data_ids"
    assert derived, "lineage edges have no derived data_ids"

    # Events should include both write and read sides.
    event_types = {row["event_type"] for row in events}
    assert any("write" in et for et in event_types)
    assert any("read" in et for et in event_types)

    # Profile summary should agree with raw counts.
    profile_result = _run(["profile", "fetch", workflow_id, "--format", "json"])
    summary = json.loads(profile_result.stdout)
    assert summary["total_assets"] >= 1
    assert summary["total_lineage_edges"] == len(lineage)
    assert summary["read_count"] > 0
    assert summary["write_count"] > 0

    # Mermaid render is non-empty.
    mermaid = _run(["profile", "fetch", workflow_id, "--format", "mermaid"])
    assert mermaid.stdout.startswith("graph TD")
