"""Whether a training task waits for its busy sticky worker."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from shared.schemas.worker import WorkerStatus
from tests.server.dispatcher.helpers import make_capturing_dispatcher


def _dispatcher(cordoned: bool) -> Any:
    disp = make_capturing_dispatcher()
    registry = cast(Any, disp._worker_registry)
    registry.get_worker.return_value = SimpleNamespace(
        id="wkr-1", status=WorkerStatus.BUSY
    )
    registry.is_worker_stale.return_value = False
    registry.is_cordoned.return_value = cordoned
    return disp


@pytest.mark.parametrize(("cordoned", "waits"), [(False, True), (True, False)])
def test_training_task_waits_only_for_an_uncordoned_sticky_worker(
    cordoned: bool, waits: bool
) -> None:
    record = SimpleNamespace(task_type="sft", category="training")
    disp = _dispatcher(cordoned)
    assert disp._should_wait_for_sticky_worker(record, "wkr-1") is waits
