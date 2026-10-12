"""A rented vast instance must tell its worker which GPUs it has.

The SSH executor slices GPUs from WORKER_HOST_GPU_ID. The docker adapter sets
it; the vast adapter did not, so `taskType: ssh, gpu.count: 1` on a vast
RTX 5080 worker failed with "requested 1 GPU(s) but this worker has none"
while the task's usage record listed the RTX 5080 (2026-10-12).
"""

from server.supervisor.adapters.vastai import _offer_gpu_environment, offer_gpu_arch


def test_single_gpu_offer_exposes_device_zero_and_arch():
    arch = offer_gpu_arch("RTX 5080")
    env = _offer_gpu_environment({"num_gpus": 1, "gpu_name": "RTX 5080"}, arch)
    assert env["WORKER_HOST_GPU_ID"] == "0"
    assert env["WORKER_HOST_GPU_ARCH"] == arch.value


def test_multi_gpu_offer_lists_every_rented_device():
    env = _offer_gpu_environment({"num_gpus": 4}, None)
    assert env == {"WORKER_HOST_GPU_ID": "0,1,2,3"}


def test_gpuless_or_unknown_offer_sets_nothing():
    assert _offer_gpu_environment({"num_gpus": 0}, None) == {}
    assert _offer_gpu_environment({}, None) == {}
    assert _offer_gpu_environment(None, None) == {}
    assert _offer_gpu_environment({"num_gpus": "n/a"}, None) == {}
