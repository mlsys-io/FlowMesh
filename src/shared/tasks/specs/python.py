"""Spec for ``python`` tasks: a function from ``code``, run in an isolated
container, whose return value and metrics become the task's result."""

from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from ...utils.pydantic_utils import copy_preserving_fields_set
from ...utils.redact import (
    has_redacted_credential_fields,
    redact_credential_fields,
)
from .._base import StrictBaseModel
from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase
from .ssh import SSHInputSpec

MAX_CODE_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 3600.0
OUTPUT_MOUNT_PATH = "/mnt/flowmesh/output"


class PythonOutputSpec(StrictBaseModel):
    maxBytes: int | None = Field(
        default=None, ge=0, description="Cap on the collected output directory size."
    )


def _validate[T: "PythonSpecStrict | PythonSpecTemplate"](spec: T) -> T:
    if not spec.code.strip():
        raise ValueError("python spec.code must be non-empty")
    if len(spec.code.encode()) > MAX_CODE_BYTES:
        raise ValueError(
            f"python spec.code is {len(spec.code.encode())} bytes; "
            f"the limit is {MAX_CODE_BYTES}"
        )
    if not spec.entrypoint.isidentifier():
        raise ValueError(
            f"python spec.entrypoint {spec.entrypoint!r} is not a Python identifier"
        )
    if spec.timeoutSeconds is not None and not (
        0 < spec.timeoutSeconds <= MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"python spec.timeoutSeconds must be in (0, {MAX_TIMEOUT_SECONDS:g}]"
        )
    if spec.requirements and spec.network != "bridge":
        raise ValueError(
            "python spec.requirements are installed with pip and need "
            "network: bridge; to run offline, use an image that already "
            "contains them"
        )
    for name in spec.emits or []:
        if not name.strip():
            raise ValueError("python spec.emits entries must be non-empty")

    inputs = spec.inputs or []
    stages = [entry.stage.strip() for entry in inputs]
    if any(not s for s in stages):
        raise ValueError("python inputs[].stage must be non-empty")
    if len(set(stages)) != len(stages):
        raise ValueError("python inputs[].stage must be unique")
    mounts = [e.mountPath.strip() for e in inputs if e.mountPath is not None]
    if any(not m for m in mounts) or len(set(mounts)) != len(mounts):
        raise ValueError("python inputs[].mountPath must be non-empty and unique")
    output = PurePosixPath(OUTPUT_MOUNT_PATH)
    for mount in mounts:
        path = PurePosixPath(mount)
        if path == output or output in path.parents:
            raise ValueError(
                f"python inputs[].mountPath {mount!r} would mount over the "
                f"task's output directory {OUTPUT_MOUNT_PATH}"
            )
    if spec.dependsOn:
        declared = {d.strip() for d in spec.dependsOn if d.strip()}
        missing = sorted(s for s in stages if s not in declared)
        if missing:
            raise ValueError(
                "python inputs must reference declared dependsOn stages; "
                f"missing: {', '.join(missing)}"
            )
    return spec


class PythonSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.PYTHON]

    placeholder_exempt_fields: ClassVar[frozenset[str]] = frozenset({"code"})

    code: str = Field(
        description="Python source defining the entrypoint function, at most "
        "256 KiB. Taken verbatim: ${...} in it is not a stage reference.",
    )
    entrypoint: str = Field(
        default="main",
        description="Function to call. Each parameter named after an input "
        "stage receives that stage's output; a parameter named inputs "
        "receives every input stage.",
    )
    image: str | None = Field(
        default=None,
        description="Container image; must provide python3. Defaults to "
        "python:3.12-slim.",
    )
    requirements: list[str] | None = Field(
        default=None,
        description="pip requirement specifiers installed before the call; "
        "needs network: bridge.",
    )
    inputs: list[SSHInputSpec] | None = Field(
        default=None,
        description="Upstream stages to mount. Omitted: each direct dependency, "
        "at /mnt/flowmesh/inputs/<stage>.",
    )
    timeoutSeconds: float | None = Field(
        default=None,
        description="Wall-clock limit, default 600 and at most 3600; the "
        "worker's SSH_MAX_TTL_SEC also caps it. Reaching it fails the task.",
    )
    network: Literal["none", "bridge"] = Field(
        default="none",
        description="none: no network at all. bridge: the worker's isolated "
        "session network.",
    )
    env: dict[str, Any] | None = Field(
        default=None, description="Extra environment variables for the code."
    )
    emits: list[str] | None = Field(
        default=None,
        description="Metric names the code promises to report; the task fails "
        "if one is missing, so an experiment never records a silent zero.",
    )
    pythonOutput: PythonOutputSpec | None = Field(
        default=None, description="Limits on the collected output directory."
    )

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"env": redact_credential_fields(spec.env)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.env
        )

    @model_validator(mode="after")
    def _check(self) -> "PythonSpecStrict":
        return _validate(self)

    def uses_gpu(self) -> bool:
        resources = self.resources
        hardware = resources.hardware if resources is not None else None
        gpu = hardware.gpu if hardware is not None else None
        return gpu is not None and gpu.count != 0


class PythonSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.PYTHON]

    placeholder_exempt_fields: ClassVar[frozenset[str]] = frozenset({"code"})

    code: str
    entrypoint: str = "main"
    image: str | None = None
    requirements: list[str] | None = None
    inputs: list[SSHInputSpec] | None = None
    timeoutSeconds: float | None = None
    network: Literal["none", "bridge"] = "none"
    env: dict[str, Any] | None = None
    emits: list[str] | None = None
    pythonOutput: PythonOutputSpec | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"env": redact_credential_fields(spec.env)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.env
        )

    @model_validator(mode="after")
    def _check(self) -> "PythonSpecTemplate":
        return _validate(self)

    def uses_gpu(self) -> bool:
        resources = self.resources
        hardware = resources.hardware if resources is not None else None
        gpu = hardware.gpu if hardware is not None else None
        return gpu is not None and gpu.count != 0
