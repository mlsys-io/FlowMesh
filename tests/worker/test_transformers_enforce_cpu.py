# ruff: noqa: E402
from unittest.mock import patch

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed (needs --extra inference)"
)

from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.transformers_executor import HFTransformersExecutor


@pytest.fixture
def executor() -> HFTransformersExecutor:
    return HFTransformersExecutor(DEFAULT_WORKER_CONFIG)


@pytest.fixture
def cuda_available():
    with patch.object(torch.cuda, "is_available", return_value=True):
        yield


@pytest.mark.usefixtures("cuda_available")
class TestEnforceCpu:
    @pytest.mark.parametrize(
        "cfg", [{}, {"device_map": "auto"}, {"device_map": "cuda"}]
    )
    def test_enforce_cpu_wins(
        self, executor: HFTransformersExecutor, cfg: dict
    ) -> None:
        assert executor._pick_device(cfg, enforce_cpu=True) == "cpu"

    def test_without_enforce_cpu_a_gpu_is_still_preferred(
        self, executor: HFTransformersExecutor
    ) -> None:
        assert executor._pick_device({}) == "cuda"

    def test_device_map_still_honoured(self, executor: HFTransformersExecutor) -> None:
        assert executor._pick_device({"device_map": "auto"}) == "auto"
        assert executor._pick_device({"device_map": "cpu"}) == "cpu"


class TestEnforceCpuWithoutGpu:
    def test_enforce_cpu_is_a_noop_when_no_gpu_exists(
        self, executor: HFTransformersExecutor
    ) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert executor._pick_device({}, enforce_cpu=True) == "cpu"
            assert executor._pick_device({}) == "cpu"
