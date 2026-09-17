"""Concurrency tests for the on-disk result write path."""

import json
import threading
from pathlib import Path

from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.schemas.result.io import read_result, result_file_path, write_result

TASK_ID = "tsk-3252724d-e271-46e8-9436-9976f7c62143"


def test_concurrent_write_result_for_same_task_does_not_raise(tmp_path: Path) -> None:
    """Regression for the office-cluster 500 on result delivery.

    ``POST /v1/results`` runs on the event loop while the ``tasks-events``
    thread mirrors merged-child results onto the same volume. Both materialize
    ``<results>/<task_id>/``, and the check-then-create in ``prepare_output_dir``
    used to let the loser raise
    ``[Errno 17] File exists: '/mnt/flowmesh-results/<task_id>'`` — surfaced to
    the worker as ``HTTP delivery ... returned status 500`` and failing the task.
    """
    threads = 8
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def deliver(index: int) -> None:
        envelope = ResultEnvelope(
            task_id=TASK_ID,
            result=BaseExecutorResult(),
            worker_id=f"wkr-{index}",
        )
        barrier.wait()
        try:
            write_result(tmp_path, envelope)
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            with lock:
                errors.append(exc)

    pool = [threading.Thread(target=deliver, args=(i,)) for i in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()

    assert errors == []

    # Every writer raced on one path, so exactly one envelope survives intact —
    # os.replace is atomic, so the file is never a torn mix of two writers.
    stored = json.loads(read_result(tmp_path, TASK_ID))
    assert stored["task_id"] == TASK_ID
    assert stored["worker_id"] in {f"wkr-{i}" for i in range(threads)}

    # No temp files from the atomic writes are left behind.
    parent = result_file_path(tmp_path, TASK_ID).parent
    assert sorted(p.name for p in parent.iterdir()) == [
        "artifacts",
        "logs",
        "results.json",
    ]
