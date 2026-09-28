"""The python task in a REAL container — opt-in, needs a local Docker daemon.

    FLOWMESH_DOCKER_TESTS=1 pytest tests/worker/test_python_executor_docker.py

The unit tests prove the executor asks for isolation; these prove the daemon
delivers it: no network, an unprivileged uid, the memory cap, the timeout, and
that a result and its metrics come back out of the container, and that
SSH_MAX_DISK stops a python or SSH task whose container layer outgrows it.
"""

import os
from pathlib import Path
from typing import Any, cast

import pytest

from shared.schemas.result import PythonResult
from shared.schemas.worker import SSHLimits
from shared.tasks.specs import PythonSpecStrict, SSHSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_live_worker_config, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.python_executor import PythonExecutor
from worker.executors.run_control import RunControl
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.utils.docker import docker_available

pytestmark = pytest.mark.skipif(
    os.getenv("FLOWMESH_DOCKER_TESTS") != "1" or not docker_available(),
    reason="set FLOWMESH_DOCKER_TESTS=1 with a reachable Docker daemon",
)


def _run(
    tmp_path: Path, code: str, upstream: dict[str, str] | None = None, **spec: object
) -> PythonResult:
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
        return executor.run(task, out, RunControl(task.task_id))
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
        _run(tmp_path, "import os\ndef main():\n    os._exit(124)\n")


def test_sys_exit_is_a_failure(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match=r"called sys.exit\(0\)"):
        _run(tmp_path, "import sys\ndef main():\n    sys.exit(0)\n")


def test_upstream_results_are_mounted_by_default(tmp_path: Path) -> None:
    results = tmp_path / "worker-results" / "t-up"
    (results / "artifacts").mkdir(parents=True)
    (results / "results.json").write_text(
        '{"task_id": "t-up", "result": {"items": [{"output": "hi"}]}}'
    )
    code = (
        "import json, os\n"
        "def main(inputs):\n"
        "    path = os.path.join(inputs['prep'], 'results.json')\n"
        "    return json.load(open(path))['result']['items'][0]['output']\n"
    )
    result = _run(tmp_path, code, upstream={"prep": "t-up"})
    assert result.value == "hi"


def test_upstream_outputs_bind_by_parameter_name(tmp_path: Path) -> None:
    upstream = tmp_path / "worker-results" / "t-py"
    (upstream / "artifacts").mkdir(parents=True)
    (upstream / "results.json").write_text(
        '{"task_id": "t-py", "result": {"ok": true, "task_type": "python", '
        '"exit_code": 0, "value": {"rows": [1, 2, 3]}}}'
    )
    code = "def main(prep):\n    return sum(prep['rows'])\n"
    result = _run(tmp_path, code, upstream={"prep": "t-py"})
    assert result.value == 6


def test_multiprocessing_runs_the_codes_functions(tmp_path: Path) -> None:
    code = (
        "import concurrent.futures, multiprocessing\n"
        "def square(n):\n"
        "    return n * n\n"
        "def main():\n"
        "    with multiprocessing.get_context('spawn').Pool(2) as pool:\n"
        "        spawned = pool.map(square, [1, 2, 3])\n"
        "    with concurrent.futures.ProcessPoolExecutor(2) as ex:\n"
        "        forked = list(ex.map(square, [4, 5]))\n"
        "    return spawned + forked\n"
    )
    assert _run(tmp_path, code).value == [1, 4, 9, 16, 25]


def test_planted_symlinks_never_reach_the_worker(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "def main():\n"
        "    out = os.environ['FLOWMESH_OUTPUT']\n"
        "    os.symlink('/etc/hostname', os.path.join(out, 'leak'))\n"
        "    open(os.path.join(out, 'kept.txt'), 'w').write('ok')\n"
        "    return 1\n"
    )
    assert _run(tmp_path, code).value == 1
    artifacts = tmp_path / "out" / "artifacts"
    assert (artifacts / "kept.txt").read_text() == "ok"
    assert not os.path.lexists(artifacts / "leak")


def test_output_past_max_bytes_is_refused_at_collection(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "def main():\n"
        "    path = os.path.join(os.environ['FLOWMESH_OUTPUT'], 'big.bin')\n"
        "    open(path, 'wb').write(b'x' * 2_000_000)\n"
        "    return 1\n"
    )
    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        _run(tmp_path, code, pythonOutput={"maxBytes": 1_000_000})
    assert not (tmp_path / "out" / "artifacts" / "big.bin").exists()


def test_scratch_is_tmpfs_and_the_root_stays_writable(tmp_path: Path) -> None:
    code = (
        "def main():\n"
        "    mounts = {}\n"
        "    for line in open('/proc/mounts'):\n"
        "        _, target, fstype, opts = line.split()[:4]\n"
        "        mounts[target] = (fstype, opts.split(',')[0])\n"
        "    scratch = ['/tmp', '/var/tmp', '/run/lock', '/dev/shm']\n"
        "    return {'scratch': [mounts.get(p, ['?'])[0] for p in scratch],\n"
        "            'root': mounts['/'][1]}\n"
    )
    value = _run(tmp_path, code).value
    assert value == {"scratch": ["tmpfs"] * 4, "root": "rw"}


def _run_with_disk_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    cap: int,
    upstream: dict[str, str] | None = None,
    **worker: Any,
) -> PythonResult:
    monkeypatch.setenv("SSH_POLL_INTERVAL_SEC", "0.5")
    task_spec = cast(
        PythonSpecStrict,
        PythonSpecStrict.model_validate({"taskType": "python", "code": code}),
    )
    executor = PythonExecutor(
        make_live_worker_config(
            tmp_path,
            ssh_network_name="flowmesh_py_test",
            ssh_limits=SSHLimits(max_disk_bytes=cap),
            **worker,
        )
    )
    task = make_worker_task_message(
        task_spec, task_type=TaskType.PYTHON, upstream_task_ids=upstream
    )
    out = tmp_path / "out"
    out.mkdir()
    try:
        return executor.run(task, out, RunControl(task.task_id))
    finally:
        executor.teardown()


def test_output_past_the_disk_cap_stops_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$FLOWMESH_OUTPUT is in the container layer, so SSH_MAX_DISK covers it."""
    code = (
        "import os, time\n"
        "def main():\n"
        "    path = os.path.join(os.environ['FLOWMESH_OUTPUT'], 'big.bin')\n"
        "    open(path, 'wb').write(b'x' * 64_000_000)\n"
        "    time.sleep(30)\n"
    )
    with pytest.raises(ExecutionError, match="disk usage exceeded"):
        _run_with_disk_cap(tmp_path, monkeypatch, code, cap=16 * 1024**2)


def test_tmpfs_scratch_does_not_count_against_the_disk_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = (
        "import time\n"
        "def main():\n"
        "    for d in ('/tmp', '/var/tmp', '/run/lock'):\n"
        "        open(d + '/scratch.bin', 'wb').write(b'x' * 32_000_000)\n"
        "    time.sleep(2)\n"
        "    return 'ok'\n"
    )
    result = _run_with_disk_cap(tmp_path, monkeypatch, code, cap=16 * 1024**2)
    assert result.value == "ok"


def test_staged_inputs_do_not_count_against_the_disk_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker with a results volume copies inputs into the container layer."""
    results = tmp_path / "worker-results"
    upstream = results / "t-up"
    (upstream / "artifacts").mkdir(parents=True)
    (upstream / "artifacts" / "big.bin").write_bytes(b"x" * 32_000_000)
    (upstream / "results.json").write_text('{"task_id": "t-up", "result": {}}')
    code = (
        "import os\n"
        "def main(inputs):\n"
        "    path = os.path.join(inputs['prep'], 'artifacts', 'big.bin')\n"
        "    return os.path.getsize(path)\n"
    )
    result = _run_with_disk_cap(
        tmp_path,
        monkeypatch,
        code,
        cap=16 * 1024**2,
        upstream={"prep": "t-up"},
        results_mount_source=results.as_posix(),
    )
    assert result.value == 32_000_000


def test_ssh_task_past_the_disk_cap_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same watchdog bounds a plain non-interactive SSH task."""
    monkeypatch.setenv("SSH_POLL_INTERVAL_SEC", "0.5")
    spec = SSHSpecStrict.model_validate(
        {
            "taskType": "ssh",
            "interactive": False,
            "image": "python:3.12-slim",
            "command": [
                "sh",
                "-c",
                "head -c 64000000 /dev/zero > /root/big.bin; sleep 30",
            ],
            "ttlSeconds": 60,
        }
    )
    executor = SSHExecutor(
        make_live_worker_config(
            tmp_path,
            ssh_network_name="flowmesh_py_test",
            ssh_limits=SSHLimits(max_disk_bytes=16 * 1024**2),
        )
    )
    task = make_worker_task_message(spec, task_type=TaskType.SSH)
    out = tmp_path / "out"
    out.mkdir()
    try:
        with pytest.raises(ExecutionError, match="disk usage exceeded"):
            executor.run(task, out, RunControl(task.task_id))
    finally:
        executor.teardown()
