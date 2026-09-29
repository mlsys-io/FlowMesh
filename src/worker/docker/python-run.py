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

Contract with the caller's code: the entrypoint is called with keyword arguments
bound by parameter name. A parameter named after an input stage receives that
stage's output — a python stage's return value, ``None`` for a stage skipped by
its condition, and any other stage's result object — and ``inputs`` receives
every input stage as a ``StageInput``; ``**kwargs`` collects the stages not
bound otherwise. The entrypoint returns a JSON-serialisable value, written to
``result.json``. Metrics come from
a ``"metrics"`` mapping in that value and/or a ``metrics.json`` the code writes
into ``$FLOWMESH_OUTPUT`` itself; both are merged into ``metrics.json`` and must
be finite numbers. ``requirements`` are installed after the switch to the
unprivileged uid, into a directory under ``/tmp`` put on ``sys.path``. Any
failure — an exception, a non-serialisable value, a promised metric that never
arrived — writes ``error.json`` and exits non-zero, so the task fails loudly.
"""

import functools
import inspect
import json
import math
import numbers
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path

OUT = os.environ.get("FLOWMESH_PY_OUTPUT", "/mnt/flowmesh/output")
DEPS = "/tmp/flowmesh-deps"  # nosec B108 - inside the task container; /tmp is its private tmpfs


def _write(name, payload, lenient=False):
    # Strict by default: a return value json cannot encode is the caller's bug
    # and must fail the task, not be stringified into a result that looks fine.
    text = json.dumps(
        payload, indent=2, default=str if lenient else None, allow_nan=lenient
    )
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


def _fail_from(exc, entrypoint):
    # A task ends only by returning: sys.exit() in the caller's code, whatever
    # its code, would otherwise skip the result and the promised metrics.
    if isinstance(exc, SystemExit):
        _fail(
            "SystemExit",
            f"the code called sys.exit({exc.code!r}); return from "
            f"{entrypoint!r} instead",
        )
    _fail(type(exc).__name__, str(exc), 1, traceback.format_exc())


class InputError(Exception):
    pass


class StageInput(os.PathLike):
    """One upstream stage, mounted read-only at ``path``.

    Usable wherever a path is (``os.path.join``, ``open``, ``Path``). Its
    ``results.json`` is read on first access to ``output``, ``result``,
    ``metadata``, ``skipped`` or ``task_type``, and a missing or unreadable one
    raises ``InputError``.
    """

    def __init__(self, stage, path):
        self.stage = stage
        self.path = Path(path)
        self.artifacts = self.path / "artifacts"

    def __fspath__(self):
        return str(self.path)

    def __repr__(self):
        return f"StageInput({self.stage!r}, {str(self.path)!r})"

    @functools.cached_property
    def _envelope(self):
        file = self.path / "results.json"
        try:
            with open(file) as fh:
                envelope = json.load(fh)
        except (OSError, ValueError) as exc:
            raise InputError(f"stage {self.stage!r}: cannot read {file}: {exc}")
        if not isinstance(envelope, dict) or not isinstance(
            envelope.get("result"), dict
        ):
            raise InputError(f"stage {self.stage!r}: {file} holds no result")
        return envelope

    @functools.cached_property
    def result(self):
        # _artifacts carries the producing worker's host path, which means
        # nothing inside this container; use ``artifacts`` / ``artifact()``.
        return {k: v for k, v in self._envelope["result"].items() if k != "_artifacts"}

    @property
    def metadata(self):
        return self._envelope.get("metadata") or {}

    @property
    def skipped(self):
        return bool(self.metadata.get("skipped"))

    @property
    def task_type(self):
        return self.result.get("task_type")

    @property
    def output(self):
        if self.skipped:
            return None
        if self.task_type == "python":
            return self.result.get("value")
        return self.result

    def artifact(self, ref):
        """Path of an artifact, from a ``{"path": ...}`` ref or a relative path."""
        relative = ref.get("path") if isinstance(ref, dict) else ref
        if not isinstance(relative, str) or not relative:
            raise InputError(
                f"stage {self.stage!r}: not an artifact reference: {ref!r}"
            )
        root = self.artifacts.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise InputError(
                f"stage {self.stage!r}: artifact {relative!r} is outside its artifacts"
            )
        return target


def _bind(fn, inputs):
    """Keyword arguments for ``fn``, bound from ``inputs`` by parameter name."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError) as exc:
        _fail("EntrypointError", f"cannot read the entrypoint's signature: {exc}", 3)
    kwargs = {}
    collect_rest = False
    available = ", ".join(inputs) or "none"
    for param in params:
        if param.kind is param.VAR_KEYWORD:
            collect_rest = True
        elif param.kind in (param.VAR_POSITIONAL, param.POSITIONAL_ONLY):
            _fail(
                "EntrypointError",
                f"parameter {param.name!r} cannot be bound by name; inputs are "
                f"passed as keyword arguments (inputs: {available})",
                3,
            )
        elif param.name == "inputs":
            kwargs["inputs"] = inputs
        elif param.name in inputs:
            kwargs[param.name] = inputs[param.name].output
        elif param.default is param.empty:
            _fail(
                "EntrypointError",
                f"parameter {param.name!r} matches no input stage "
                f"(inputs: {available})",
                3,
            )
    if collect_rest:
        for stage, stage_input in inputs.items():
            if stage not in kwargs:
                kwargs[stage] = stage_input.output
    return kwargs


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
    inputs = {
        stage: StageInput(stage, path)
        for stage, path in json.loads(
            os.environ.get("FLOWMESH_PY_INPUTS") or "{}"
        ).items()
    }
    emits = json.loads(os.environ.get("FLOWMESH_PY_EMITS") or "[]")

    # A registered, importable module, so what the code defines can be pickled
    # (multiprocessing, concurrent.futures) and re-imported by spawned children.
    module = types.ModuleType(Path(code_path).stem)
    module.__file__ = code_path
    sys.modules[module.__name__] = module
    sys.path.insert(0, os.path.dirname(code_path))
    try:
        with open(code_path) as fh:
            exec(compile(fh.read(), code_path, "exec"), module.__dict__)
    except BaseException as exc:
        _fail_from(exc, entrypoint)
    fn = getattr(module, entrypoint, None)
    if not callable(fn):
        _fail("EntrypointError", f"the code defines no function {entrypoint!r}", 3)
    try:
        kwargs = _bind(fn, inputs)
    except InputError as exc:
        _fail("InputError", str(exc), 3)
    try:
        value = fn(**kwargs)
    except BaseException as exc:
        _fail_from(exc, entrypoint)

    metrics = _collect_metrics(value)
    try:
        _write("result.json", value)
    except Exception as exc:
        _fail("ResultError", f"return value is not JSON-serialisable: {exc}", 3)
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
