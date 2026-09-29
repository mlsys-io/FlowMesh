"""The python task in a REAL container — opt-in, needs a local Docker daemon.

    FLOWMESH_DOCKER_TESTS=1 pytest tests/worker/test_python_executor_docker.py

The unit tests prove the executor asks for isolation; these prove the daemon
delivers it: no network, an unprivileged uid, the memory cap, the timeout, and
that a result and its metrics come back out of the container.
"""

import os
from pathlib import Path
from typing import cast

import pytest

from shared.tasks.specs import PythonSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_live_worker_config, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.python_executor import PythonExecutor
from worker.executors.utils.docker import docker_available

pytestmark = pytest.mark.skipif(
    os.getenv("FLOWMESH_DOCKER_TESTS") != "1" or not docker_available(),
    reason="set FLOWMESH_DOCKER_TESTS=1 with a reachable Docker daemon",
)


def _run(  # type: ignore[no-untyped-def]
    tmp_path: Path, code: str, upstream: dict[str, str] | None = None, **spec: object
):
    task_spec = cast(
        PythonSpecStrict,
        PythonSpecStrict.model_validate({"taskType": "python", "code": code, **spec}),
    )
    executor = PythonExecutor(
        make_live_worker_config(tmp_path, ssh_network_name="flowmesh_py_test")
    )
    task = make_worker_task_message(
        task_spec, task_type=TaskType.PYTHON, upstream_task_ids=upstream
    )
    out = tmp_path / "out"
    out.mkdir()
    try:
        return executor.run(task, out)
    finally:
        executor.teardown()


def test_result_and_metrics_come_back(tmp_path: Path) -> None:
    code = "def main():\n    return {'answer': 42, 'metrics': {'score': 0.75}}\n"
    result = _run(tmp_path, code, emits=["score"])
    assert result.exit_code == 0
    assert result.value == {"answer": 42, "metrics": {"score": 0.75}}
    assert result.metrics == {"score": 0.75}


def test_runs_unprivileged_and_offline(tmp_path: Path) -> None:
    code = (
        "import os, socket\n"
        "def main():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "        net = 'reachable'\n"
        "    except OSError as exc:\n"
        "        net = type(exc).__name__\n"
        "    return {'uid': os.getuid(), 'net': net}\n"
    )
    result = _run(tmp_path, code)
    assert result.value["uid"] == 65534
    assert result.value["net"] != "reachable"


def test_no_effective_capabilities(tmp_path: Path) -> None:
    code = (
        "def main():\n"
        "    status = open('/proc/self/status').read()\n"
        "    return status.split('CapEff:')[1].split()[0]\n"
    )
    assert _run(tmp_path, code).value == "0000000000000000"


def test_callers_exception_is_the_failure_message(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="ValueError: bad input"):
        _run(tmp_path, "def main():\n    raise ValueError('bad input')\n")


def test_timeout_fails(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="timed out after 3s"):
        _run(
            tmp_path, "import time\ndef main():\n    time.sleep(60)\n", timeoutSeconds=3
        )


def test_memory_cap_is_enforced(tmp_path: Path) -> None:
    code = "def main():\n    x = bytearray(1024 * 1024 * 1024)\n    return len(x)\n"
    with pytest.raises(ExecutionError, match="exit 137"):
        _run(tmp_path, code, resources={"hardware": {"memory": "128Mi"}})


def test_requirements_install_over_bridge(tmp_path: Path) -> None:
    code = "import six, os\ndef main():\n    return [six.__version__, os.getuid()]\n"
    result = _run(tmp_path, code, requirements=["six==1.16.0"], network="bridge")
    assert result.value == ["1.16.0", 65534]


def test_binary_requirements_install_and_load(tmp_path: Path) -> None:
    # A compiled extension: loading it needs an executable /tmp.
    code = (
        "import orjson, os\n"
        "def main():\n"
        "    return [orjson.dumps(1).decode(), os.getuid()]\n"
    )
    result = _run(tmp_path, code, requirements=["orjson==3.10.18"], network="bridge")
    assert result.value == ["1", 65534]


def test_no_gpus_unless_asked(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "def main():\n"
        "    return [os.environ.get('NVIDIA_VISIBLE_DEVICES'),\n"
        "            os.environ.get('CUDA_VISIBLE_DEVICES')]\n"
    )
    assert _run(tmp_path, code).value == ["void", None]


def test_exit_code_124_is_not_a_timeout(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="exited with code 124"):
        _run(tmp_path, "import sys\ndef main():\n    sys.exit(124)\n")


def test_upstream_results_are_mounted_by_default(tmp_path: Path) -> None:
    results = tmp_path / "worker-results" / "t-up"
    results.mkdir(parents=True)
    (results / "results.json").write_text('{"result": {"items": [{"output": "hi"}]}}')
    code = (
        "import json, os\n"
        "def main(inputs):\n"
        "    path = os.path.join(inputs['prep'], 'results.json')\n"
        "    return json.load(open(path))['result']['items'][0]['output']\n"
    )
    result = _run(tmp_path, code, upstream={"prep": "t-up"})
    assert result.value == "hi"
