"""FlowMesh python-task bootstrap. Runs inside the task container.

Invoked by the non-interactive entrypoint wrapper (ssh-run.sh) after it has
staged the upstream inputs, as ``python3 /opt/flowmesh/python-run.py``. Stdlib
only: the image is the caller's, so nothing else can be assumed present.

Environment (set by worker/executors/python_executor.py):
  FLOWMESH_PY_CODE          path of the caller's source file
  FLOWMESH_PY_ENTRYPOINT    function to call
  FLOWMESH_PY_INPUTS        JSON {stage: mounted directory}
  FLOWMESH_PY_OUTPUT        directory collected as the task's artifacts
  FLOWMESH_PY_REQUIREMENTS  JSON list of pip specifiers (network: bridge only)
  FLOWMESH_PY_EMITS         JSON list of metric names that must be reported
  FLOWMESH_PY_UID           uid/gid to run the caller's code as

Contract with the caller's code: ``entrypoint(inputs)`` (or ``entrypoint()``)
returns a JSON-serialisable value, written to ``result.json``. Metrics come from
a ``"metrics"`` mapping in that value and/or a ``metrics.json`` the code writes
into ``$FLOWMESH_OUTPUT`` itself; both are merged into ``metrics.json`` and must
be finite numbers. ``requirements`` are installed after the switch to the
unprivileged uid, into a directory under ``/tmp`` put on ``sys.path``. Any
failure — an exception, a non-serialisable value, a promised metric that never
arrived — writes ``error.json`` and exits non-zero, so the task fails loudly.
"""

import inspect
import json
import math
import numbers
import os
import subprocess
import sys
import traceback

OUT = os.environ.get("FLOWMESH_PY_OUTPUT", "/mnt/flowmesh/output")
DEPS = "/tmp/flowmesh-deps"  # nosec B108 - inside the task container; /tmp is its private tmpfs


def _write(name, payload, lenient=False):
    # Strict by default: a return value json cannot encode is the caller's bug
    # and must fail the task, not be stringified into a result that looks fine.
    text = json.dumps(payload, indent=2, default=str if lenient else None)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)


def _fail(kind, message, exit_code=1, tb=None):
    try:
        _write(
            "error.json",
            {"type": kind, "message": message, "traceback": tb},
            lenient=True,
        )
    except Exception:
        pass
    print(f"flowmesh python task failed: {kind}: {message}", file=sys.stderr)
    if tb:
        print(tb, file=sys.stderr)
    sys.exit(exit_code)


def _install_requirements():
    reqs = json.loads(os.environ.get("FLOWMESH_PY_REQUIREMENTS") or "[]")
    if not reqs:
        return
    cmd = [
        *(sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet"),
        "--disable-pip-version-check",
    ]
    proc = subprocess.run(
        [*cmd, "--target", DEPS, *reqs]
    )  # nosec B603 - argv list, no shell, the task owner's own requirements in their own container
    if proc.returncode != 0:
        _fail("RequirementsError", f"pip install {' '.join(reqs)} failed", 2)
    sys.path.insert(0, DEPS)


def _drop_privileges(uid):
    os.makedirs(OUT, exist_ok=True)
    if os.getuid() != 0:
        return
    for root, dirs, files in os.walk(OUT):
        for name in [root, *(os.path.join(root, n) for n in dirs + files)]:
            os.chown(name, uid, uid)
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)


def _collect_metrics(value):
    metrics = {}
    written = os.path.join(OUT, "metrics.json")
    if os.path.exists(written):
        try:
            with open(written) as fh:
                loaded = json.load(fh)
        except (OSError, ValueError) as exc:
            _fail("MetricsError", f"metrics.json is not valid JSON: {exc}", 3)
        if not isinstance(loaded, dict):
            _fail("MetricsError", "metrics.json must hold a JSON object", 3)
        metrics.update(loaded)
    if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
        metrics.update(value["metrics"])
    for name, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            _fail("MetricsError", f"metric {name!r} is not a number: {v!r}", 3)
        if not _finite(v):
            _fail("MetricsError", f"metric {name!r} is not finite: {v!r}", 3)
    return {k: float(v) for k, v in metrics.items()}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def main():
    uid = int(os.environ.get("FLOWMESH_PY_UID", "65534"))
    os.makedirs(OUT, exist_ok=True)
    _drop_privileges(uid)
    os.environ["HOME"] = "/tmp"  # nosec B108 - the container's private tmpfs
    os.environ["FLOWMESH_OUTPUT"] = OUT
    os.chdir("/tmp")  # nosec B108 - the container's private tmpfs
    _install_requirements()

    code_path = os.environ["FLOWMESH_PY_CODE"]
    entrypoint = os.environ.get("FLOWMESH_PY_ENTRYPOINT", "main")
    inputs = json.loads(os.environ.get("FLOWMESH_PY_INPUTS") or "{}")
    emits = json.loads(os.environ.get("FLOWMESH_PY_EMITS") or "[]")

    namespace = {"__name__": "__flowmesh_task__", "__file__": code_path}
    try:
        with open(code_path) as fh:
            exec(compile(fh.read(), "task.py", "exec"), namespace)
        fn = namespace.get(entrypoint)
        if not callable(fn):
            _fail("EntrypointError", f"task.py defines no function {entrypoint!r}", 3)
        takes_args = bool(inspect.signature(fn).parameters)
        value = fn(inputs) if takes_args else fn()
    except SystemExit:
        raise
    except BaseException as exc:
        _fail(type(exc).__name__, str(exc), 1, traceback.format_exc())

    try:
        _write("result.json", value)
    except Exception as exc:
        _fail("ResultError", f"return value is not JSON-serialisable: {exc}", 3)
    metrics = _collect_metrics(value)
    if metrics:
        _write("metrics.json", metrics)
    missing = [m for m in emits if m not in metrics]
    if missing:
        _fail(
            "MetricsError",
            f"declared emits {missing} were not reported; return "
            '{"metrics": {...}} or write $FLOWMESH_OUTPUT/metrics.json',
            3,
        )


if __name__ == "__main__":
    main()
