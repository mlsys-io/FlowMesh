from typing import Any, Literal, Self

from pydantic import model_validator

from ...utils.pydantic_utils import copy_preserving_fields_set
from ...utils.redact import (
    contains_redacted,
    has_redacted_credential_fields,
    redact_credential,
    redact_credential_fields,
)
from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import TemplateInt
from ..task_type import TaskType
from .common import TaskSpecStrictBase, TaskSpecTemplateBase


class SSHInputSpec(StrictBaseModel):
    stage: str
    mountPath: str | None = None


class SSHOutputSpec(StrictBaseModel):
    mountPath: str | None = None
    maxBytes: int | None = None


class SSHOutputSpecTemplate(TemplateBaseModel):
    mountPath: str | None = None
    maxBytes: TemplateInt | None = None


def _resolve_interactive[T: "SSHSpecStrict | SSHSpecTemplate"](spec: T) -> T:
    """Resolve the ``interactive`` field when the caller left it as ``None``.

    Inference rules (applied only when ``interactive is None``):
    * If ``command`` or ``entrypoint`` is set → ``False`` (non-interactive).
    * If ``authorizedKeys`` is non-empty → ``True`` (interactive SSH session).
    * Otherwise → ``False``.

    Validation:
    * ``interactive=True`` with ``command`` or ``entrypoint`` set is rejected.
    """
    interactive = spec.interactive
    has_command = spec.command is not None or spec.entrypoint is not None

    if interactive is None:
        spec.interactive = bool((not has_command) and spec.authorizedKeys)
    elif interactive:
        if has_command:
            raise ValueError(
                "SSH spec has interactive=true but command/entrypoint are set; "
                "set interactive=false or omit the interactive field"
            )
        if not spec.authorizedKeys:
            raise ValueError(
                "Interactive SSH tasks require at least one entry in authorizedKeys"
            )

    return spec


def _validate_inputs[T: "SSHSpecStrict | SSHSpecTemplate"](spec: T) -> T:
    inputs = spec.inputs or []
    seen_stages: set[str] = set()
    seen_mount_paths: set[str] = set()

    for entry in inputs:
        stage_name = entry.stage.strip()
        # Stage name must be non-empty and unique
        if not stage_name:
            raise ValueError("SSH inputs[].stage must be non-empty")
        if stage_name in seen_stages:
            raise ValueError(f"Duplicate SSH input stage '{stage_name}'")
        seen_stages.add(stage_name)

        if entry.mountPath is not None:
            # mountPath must be non-empty and unique when set
            mount_path = entry.mountPath.strip()
            if not mount_path:
                raise ValueError("SSH inputs[].mountPath must be non-empty when set")
            if mount_path in seen_mount_paths:
                raise ValueError(f"Duplicate SSH input mountPath '{mount_path}'")
            seen_mount_paths.add(mount_path)

    output = spec.sshOutput
    if output is not None and output.mountPath is not None:
        # mountPath must be non-empty and unique when set
        output_mount_path = output.mountPath.strip()
        if not output_mount_path:
            raise ValueError("SSH sshOutput.mountPath must be non-empty when set")
        if output_mount_path in seen_mount_paths:
            raise ValueError(f"Duplicate SSH output mountPath '{output_mount_path}'")

    depends_on = spec.dependsOn
    if not depends_on:
        return spec

    # SSH inputs must reference declared dependsOn stages when dependsOn is provided
    dependency_names = {
        dep_stripped for dep in depends_on if (dep_stripped := dep.strip())
    }
    missing = sorted(stage for stage in seen_stages if stage not in dependency_names)
    if missing:
        raise ValueError(
            "SSH inputs must reference declared dependsOn stages when dependsOn is "
            f"provided; missing: {', '.join(missing)}"
        )
    return spec


class SSHSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.SSH]

    interactive: bool | None = None
    image: str | None = None
    user: str | None = None
    authorizedKeys: list[str] | None = None
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    ttlSeconds: float | None = None
    idleTimeoutSeconds: float | None = None
    accessMode: Literal["direct", "proxy", "forward"] | None = None
    inputs: list[SSHInputSpec] | None = None
    sshOutput: SSHOutputSpec | None = None
    env: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec,
            {
                "authorizedKeys": redact_credential(spec.authorizedKeys),
                "env": redact_credential_fields(spec.env),
            },
        )

    def has_redacted_credentials(self) -> bool:
        return (
            super().has_redacted_credentials()
            or contains_redacted(self.authorizedKeys)
            or has_redacted_credential_fields(self.env)
        )

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> "SSHSpecStrict":
        _resolve_interactive(self)
        _validate_inputs(self)
        return self

    def uses_gpu(self) -> bool:
        """An SSH session is GPU-using exactly when it asks for devices.

        A bare ``type`` or ``memory`` without ``count`` still resolves to one device
        in the session config, so any ``gpu`` block counts -- except an explicit
        ``count: 0``, which asks for none.
        """
        resources = self.resources
        hardware = resources.hardware if resources is not None else None
        gpu = hardware.gpu if hardware is not None else None
        return gpu is not None and gpu.count != 0


class SSHSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.SSH]

    interactive: bool | None = None
    image: str | None = None
    user: str | None = None
    authorizedKeys: list[str] | None = None
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    ttlSeconds: float | None = None
    idleTimeoutSeconds: float | None = None
    accessMode: Literal["direct", "proxy", "forward"] | None = None
    inputs: list[SSHInputSpec] | None = None
    sshOutput: SSHOutputSpecTemplate | None = None
    env: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec,
            {
                "authorizedKeys": redact_credential(spec.authorizedKeys),
                "env": redact_credential_fields(spec.env),
            },
        )

    def has_redacted_credentials(self) -> bool:
        return (
            super().has_redacted_credentials()
            or contains_redacted(self.authorizedKeys)
            or has_redacted_credential_fields(self.env)
        )

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> "SSHSpecTemplate":
        _resolve_interactive(self)
        _validate_inputs(self)
        return self

    def uses_gpu(self) -> bool:
        """An SSH session is GPU-using exactly when it asks for devices.

        A bare ``type`` or ``memory`` without ``count`` still resolves to one device
        in the session config, so any ``gpu`` block counts -- except an explicit
        ``count: 0``, which asks for none.
        """
        resources = self.resources
        hardware = resources.hardware if resources is not None else None
        gpu = hardware.gpu if hardware is not None else None
        return gpu is not None and gpu.count != 0
