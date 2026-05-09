"""Pure doctor checks shared by FlowMesh tooling."""

import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flowmesh.config import DEFAULT_CONFIG_PATH, FlowMeshConfig
from flowmesh.exceptions import ConfigInvalidError, ConfigNotFoundError

from .docker import DockerError, ensure_docker_available
from .env import validate_env_file
from .env_schema import EnvSchema, schema_keys, validate_env_values

type FindingLevel = Literal["note", "warning", "error"]

_DEFAULT_CUDA_PROBE_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu24.04"
_DEFAULT_DOCKER_GPU_RUNTIME = "nvidia"


@dataclass(frozen=True)
class DoctorFinding:
    level: FindingLevel
    message: str


@dataclass
class DoctorReport:
    findings: list[DoctorFinding] = field(default_factory=list)
    callback: Callable[[DoctorFinding], Any] | None = None

    @property
    def errors(self) -> list[str]:
        return [
            finding.message for finding in self.findings if finding.level == "error"
        ]

    @property
    def warnings(self) -> list[str]:
        return [
            finding.message for finding in self.findings if finding.level == "warning"
        ]

    @property
    def notes(self) -> list[str]:
        return [finding.message for finding in self.findings if finding.level == "note"]

    def error(self, message: str) -> None:
        self._add_finding("error", message)

    def warning(self, message: str) -> None:
        self._add_finding("warning", message)

    def note(self, message: str) -> None:
        self._add_finding("note", message)

    def extend_errors(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.error(message)

    def extend_warnings(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.warning(message)

    def extend_notes(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.note(message)

    def _add_finding(self, level: FindingLevel, message: str) -> None:
        finding = DoctorFinding(level, message)
        self.findings.append(finding)
        if self.callback:
            self.callback(finding)


def run_doctor_checks(
    env_file: Path,
    schema: EnvSchema,
    callback: Callable[[DoctorFinding], Any] | None = None,
) -> DoctorReport:
    """Run shared doctor checks and return structured findings."""
    report = DoctorReport(callback=callback)
    env_values, env_errors = validate_env_file(
        env_file, expected_keys=schema_keys(schema)
    )
    report.extend_errors(env_errors)
    if env_values is not None:
        errors, warnings = validate_env_values(schema, env_values)
        report.extend_errors(errors)
        report.extend_warnings(warnings)
    validate_config_file(report)
    validate_docker_availability(report)
    validate_gpu_visibility(report, env_values or {})
    return report


def validate_config_file(report: DoctorReport) -> None:
    """Validate the presence and basic correctness of the config file."""
    try:
        FlowMeshConfig.from_file(DEFAULT_CONFIG_PATH)
    except ConfigNotFoundError as exc:
        report.warning(str(exc))
    except ConfigInvalidError as exc:
        report.error(str(exc))
    else:
        report.note(f"Config file found at {DEFAULT_CONFIG_PATH}")


def validate_docker_availability(report: DoctorReport) -> None:
    """Validate docker CLI and daemon reachability."""
    try:
        ensure_docker_available()
    except DockerError as exc:
        report.error(str(exc))
        return
    report.note("Docker is available")
    docker_bin = _require_bin("docker")

    try:
        version = subprocess.run(
            [
                docker_bin,
                "--version",
            ],  # nosec B603: argv list, no shell, absolute path.
            capture_output=True,
            text=True,
            check=False,
        )
        if version.stdout:
            report.note(version.stdout.strip())
        elif version.stderr:
            report.note(version.stderr.strip())
    except FileNotFoundError:
        report.error("Docker CLI not found")
        return

    docker_info = subprocess.run(
        [docker_bin, "info"],  # nosec B603: argv list, no shell, absolute path.
        capture_output=True,
        text=True,
        check=False,
    )
    if docker_info.returncode == 0:
        report.note("Docker daemon: reachable")
    else:
        report.error("Docker daemon: NOT reachable")

    if shutil.which("docker-compose"):
        report.warning("docker-compose detected (legacy).")
    else:
        report.note("Using docker compose plugin.")


def validate_gpu_visibility(report: DoctorReport, env_values: dict[str, str]) -> None:
    """Validate whether GPUs are visible to the host and Docker runtime."""
    nvidia_smi_bin = shutil.which("nvidia-smi")
    if nvidia_smi_bin:
        smi = subprocess.run(
            [nvidia_smi_bin],  # nosec B603: argv list, no shell, absolute path.
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.stdout:
            report.note("nvidia-smi output:")
            report.note(smi.stdout)
        if smi.returncode != 0:
            detail = (smi.stderr or smi.stdout).strip()
            report.warning(f"nvidia-smi failed on host: {detail or 'unknown error'}")
            return
        validate_docker_gpu_runtime(report, env_values)
        return
    report.warning("nvidia-smi not found; GPU visibility not verified.")


def validate_docker_gpu_runtime(
    report: DoctorReport, env_values: dict[str, str]
) -> None:
    """Validate that the configured Docker GPU runtime works with the probe image."""
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return

    probe_image = env_values.get("SERVER_CUDA_PROBE_IMAGE", _DEFAULT_CUDA_PROBE_IMAGE)
    runtime = env_values.get("DOCKER_GPU_RUNTIME", _DEFAULT_DOCKER_GPU_RUNTIME).strip()
    command = [docker_bin, "run", "--rm"]
    if runtime:
        command += ["--runtime", runtime]
    command += [
        "--gpus",
        "all",
        probe_image,
        "nvidia-smi",
        "--query-gpu=index,name",
        "--format=csv,noheader",
    ]
    result = subprocess.run(
        command,  # nosec B603: argv list, no shell, absolute path.
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        report.note("Docker GPU probe succeeded.")
        return

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"exit code {result.returncode}"
    lowered = detail.lower()
    if runtime and "unknown or invalid runtime name" in lowered:
        report.warning(
            f"DOCKER_GPU_RUNTIME={runtime!r} is not available to Docker on this host. "
            "If `docker run --rm --gpus all ...` works without `--runtime`, set "
            "`DOCKER_GPU_RUNTIME=` in the stack env. This is common on DGX Spark."
        )
        return
    report.warning(f"Docker GPU probe failed: {detail}")


def _require_bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(name)
    return path
