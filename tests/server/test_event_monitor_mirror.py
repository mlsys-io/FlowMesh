"""``EventMonitor.mirror_task_results`` regression tests.

Cover the merge-result race where the worker's per-child
``POST /api/v1/results`` and the server's mirror-from-parent step were
both writing into the same child directory. The mirror used to ``rmtree``
+ ``copytree`` the dir even when the child already held its own result
file — the rmtree raced with the inbound POST and dropped the file,
producing ``404 result not found`` on a ``DONE`` task.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor
from server.utils.manifest import RESULTS_NAME


def _make_monitor(results_dir: Path) -> EventMonitor:
    """Build the smallest EventMonitor that exercises ``mirror_task_results``."""
    runtime = MagicMock()
    runtime.get_record.return_value = None
    return EventMonitor(
        redis_client=MagicMock(),
        stop_event=MagicMock(),
        logger=logging.getLogger("test-event-monitor"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        results_dir=results_dir,
    )


def _seed_result(results_dir: Path, task_id: str, content: dict) -> Path:
    task_dir = results_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / RESULTS_NAME
    path.write_text(json.dumps(content))
    return path


class TestMirrorTaskResults:
    def test_skips_child_already_populated_by_per_child_post(
        self, tmp_path: Path
    ) -> None:
        """The skip path preserves the child's own result file.

        When the worker POSTs a parent's combined result followed by each
        child's individual result, both lands at
        ``<results_dir>/<task_id>/results.json``. The TASK_SUCCEEDED event
        then fires ``mirror_task_results``; without the skip, it would
        ``rmtree`` the child's dir and copy the parent's content,
        clobbering the child's distinct result.
        """
        monitor = _make_monitor(tmp_path)
        _seed_result(
            tmp_path, "tsk-parent", {"task_id": "tsk-parent", "kind": "PARENT"}
        )
        child_path = _seed_result(
            tmp_path, "tsk-child", {"task_id": "tsk-child", "kind": "CHILD"}
        )

        monitor.mirror_task_results("tsk-parent", ["tsk-child"])

        # Child's own POSTed result was preserved, not overwritten.
        loaded = json.loads(child_path.read_text())
        assert loaded["task_id"] == "tsk-child"
        assert loaded["kind"] == "CHILD"

    def test_clones_parent_when_child_dir_missing(self, tmp_path: Path) -> None:
        """When the worker did not POST per-child, mirror clones from parent."""
        monitor = _make_monitor(tmp_path)
        _seed_result(
            tmp_path, "tsk-parent", {"task_id": "tsk-parent", "kind": "PARENT"}
        )

        monitor.mirror_task_results("tsk-parent", ["tsk-child"])

        child_path = tmp_path / "tsk-child" / RESULTS_NAME
        assert child_path.exists()
        loaded = json.loads(child_path.read_text())
        # The clone surfaces the parent's payload (lumilake's executor
        # leaves child-specific data inside ``parent.result.children``).
        assert loaded["kind"] == "PARENT"

    def test_defers_when_parent_dir_missing(self, tmp_path: Path) -> None:
        """Mirror is queued on ``_pending_result_clones`` if the parent
        hasn't landed yet — ``ingest_result`` drains the queue."""
        monitor = _make_monitor(tmp_path)

        monitor.mirror_task_results("tsk-parent", ["tsk-child"])

        assert "tsk-parent" in monitor._pending_result_clones
        assert monitor._pending_result_clones["tsk-parent"] == ["tsk-child"]
        # No child dir was created.
        assert not (tmp_path / "tsk-child").exists()
