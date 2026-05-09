"""Focused tests for SDK helpers and transport utilities."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from flowmesh._base_client import BaseAsyncClient, BaseClient
from flowmesh.exceptions import FlowMeshConnectionError, NotFoundError
from flowmesh.params import append_param, extend_params
from flowmesh_stack.doctor import (
    DoctorFinding,
    DoctorReport,
    run_doctor_checks,
    validate_gpu_visibility,
)
from flowmesh_stack.env_schema import EnvSchema

from .helpers import AsyncHTTP, AsyncResponse, SyncHTTP, SyncResponse
from .router_app import TEST_BASE_URL


class TestParams:
    def test_append_param_coerces_bool_and_int(self) -> None:
        params: list[tuple[str, str]] = []
        append_param(params, "enabled", True)
        append_param(params, "port", 22)
        append_param(params, "name", "ssh")
        append_param(params, "skip", None)
        assert params == [("enabled", "true"), ("port", "22"), ("name", "ssh")]

    def test_extend_params_preserves_repeated_values_in_order(self) -> None:
        params: list[tuple[str, str]] = []
        extend_params(params, "status", ["PENDING", "DONE"])
        extend_params(params, "failed", False)
        extend_params(params, "task_id", "t-1")
        assert params == [
            ("status", "PENDING"),
            ("status", "DONE"),
            ("failed", "false"),
            ("task_id", "t-1"),
        ]


class TestDoctorReport:
    def test_report_preserves_finding_order_and_views(self) -> None:
        seen: list[DoctorFinding] = []
        report = DoctorReport(callback=seen.append)
        report.note("n1")
        report.error("e1")
        report.warning("w1")
        report.extend_errors(["e2"])

        assert report.findings == [
            DoctorFinding("note", "n1"),
            DoctorFinding("error", "e1"),
            DoctorFinding("warning", "w1"),
            DoctorFinding("error", "e2"),
        ]
        assert report.notes == ["n1"]
        assert report.errors == ["e1", "e2"]
        assert report.warnings == ["w1"]
        assert seen == report.findings

    def test_report_is_safe_for_asdict_and_deepcopy(self) -> None:
        report = DoctorReport()
        report.error("boom")
        assert asdict(report)["findings"] == [{"level": "error", "message": "boom"}]
        copied = copy.deepcopy(report)
        assert copied.findings == report.findings


def test_run_doctor_checks_preserves_cross_category_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_findings: list[DoctorFinding] = []
    schema = EnvSchema(name="test", header=[], sections=[])

    monkeypatch.setattr(
        "flowmesh_stack.doctor.validate_env_file",
        lambda env_file, expected_keys: (
            {"FLOWMESH_BASE_URL": "http://localhost"},
            ["env error"],
        ),
    )
    monkeypatch.setattr(
        "flowmesh_stack.doctor.validate_env_values",
        lambda schema, env: (["schema error"], ["schema warning"]),
    )
    monkeypatch.setattr(
        "flowmesh_stack.doctor.validate_config_file",
        lambda report: report.note("config file valid"),
    )
    monkeypatch.setattr(
        "flowmesh_stack.doctor.validate_docker_availability",
        lambda report: report.error("docker down"),
    )
    monkeypatch.setattr(
        "flowmesh_stack.doctor.validate_gpu_visibility",
        lambda report, env_values: report.warning("gpu missing"),
    )

    report = run_doctor_checks(
        Path("test.env"), schema, callback=callback_findings.append
    )

    assert report.findings == [
        DoctorFinding("error", "env error"),
        DoctorFinding("error", "schema error"),
        DoctorFinding("warning", "schema warning"),
        DoctorFinding("note", "config file valid"),
        DoctorFinding("error", "docker down"),
        DoctorFinding("warning", "gpu missing"),
    ]
    assert callback_findings == report.findings


class TestDoctorGpuRuntimeChecks:
    def test_warns_when_configured_runtime_is_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = DoctorReport()

        def fake_which(name: str) -> str | None:
            mapping = {
                "nvidia-smi": "/usr/bin/nvidia-smi",
                "docker": "/usr/bin/docker",
            }
            return mapping.get(name)

        def fake_run(
            args: list[str], capture_output: bool, text: bool, check: bool
        ) -> SimpleNamespace:
            if args[0] == "/usr/bin/nvidia-smi":
                return SimpleNamespace(
                    returncode=0, stdout="0, NVIDIA GB10\n", stderr=""
                )
            if args[:6] == [
                "/usr/bin/docker",
                "run",
                "--rm",
                "--runtime",
                "nvidia",
                "--gpus",
            ]:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="docker: Error response from daemon: unknown or invalid runtime name: nvidia",
                )
            raise AssertionError(f"Unexpected command: {args}")

        monkeypatch.setattr("flowmesh_stack.doctor.shutil.which", fake_which)
        monkeypatch.setattr("flowmesh_stack.doctor.subprocess.run", fake_run)

        validate_gpu_visibility(report, {"DOCKER_GPU_RUNTIME": "nvidia"})

        assert "nvidia-smi output:" in report.notes
        assert any("DOCKER_GPU_RUNTIME='nvidia'" in warning for warning in report.warnings)
        assert any("DGX Spark" in warning for warning in report.warnings)

    def test_accepts_empty_runtime_when_docker_gpu_probe_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = DoctorReport()
        commands: list[list[str]] = []

        def fake_which(name: str) -> str | None:
            mapping = {
                "nvidia-smi": "/usr/bin/nvidia-smi",
                "docker": "/usr/bin/docker",
            }
            return mapping.get(name)

        def fake_run(
            args: list[str], capture_output: bool, text: bool, check: bool
        ) -> SimpleNamespace:
            commands.append(args)
            if args[0] == "/usr/bin/nvidia-smi":
                return SimpleNamespace(
                    returncode=0, stdout="0, NVIDIA GB10\n", stderr=""
                )
            if args[0] == "/usr/bin/docker":
                return SimpleNamespace(
                    returncode=0, stdout="0, NVIDIA GB10\n", stderr=""
                )
            raise AssertionError(f"Unexpected command: {args}")

        monkeypatch.setattr("flowmesh_stack.doctor.shutil.which", fake_which)
        monkeypatch.setattr("flowmesh_stack.doctor.subprocess.run", fake_run)

        validate_gpu_visibility(report, {"DOCKER_GPU_RUNTIME": ""})

        docker_run = next(args for args in commands if args[0] == "/usr/bin/docker")
        assert "--runtime" not in docker_run
        assert "Docker GPU probe succeeded." in report.notes


class TestBaseClientTransport:
    def test_stream_sse_parses_cursor_and_multiline_payload(self) -> None:
        response = SyncResponse(
            lines=[
                "id: cursor-1",
                "event: log",
                'data: {"message":"hello",',
                'data: "task_id":"t-1"}',
                "",
                "id: cursor-2",
                "data: plain text line",
                "",
            ]
        )
        client = BaseClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(Any, SyncHTTP(response)),
        )

        events = list(client._stream_sse("/tasks/t-1/logs/stream"))

        assert events == [
            ("cursor-1", {"message": "hello", "task_id": "t-1"}),
            ("cursor-2", {"message": "plain text line"}),
        ]

    def test_stream_sse_maps_connect_errors(self) -> None:
        request = httpx.Request("GET", f"{TEST_BASE_URL}/api/v1/tasks/t-1/logs/stream")
        client = BaseClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(
                Any,
                SyncHTTP(exc=httpx.ConnectError("boom", request=request)),
            ),
        )

        with pytest.raises(FlowMeshConnectionError):
            list(client._stream_sse("/tasks/t-1/logs/stream"))

    @pytest.mark.anyio
    async def test_async_stream_sse_parses_cursor_and_flushes_tail(self) -> None:
        client = BaseAsyncClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(
                Any,
                AsyncHTTP(
                    AsyncResponse(
                        lines=[
                            "id: cursor-1",
                            'data: {"message":"hello"}',
                            "",
                            "id: cursor-2",
                            "data: trailing text",
                        ]
                    )
                ),
            ),
        )

        events = []
        async for item in client._stream_sse("/tasks/t-1/logs/stream"):
            events.append(item)
        assert events == [
            ("cursor-1", {"message": "hello"}),
            ("cursor-2", {"message": "trailing text"}),
        ]

    @pytest.mark.anyio
    async def test_async_download_writes_bytes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        monkeypatch.setattr(
            "flowmesh._base_client.asyncio.to_thread", immediate_to_thread
        )

        client = BaseAsyncClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(
                Any,
                AsyncHTTP(
                    AsyncResponse(
                        chunks=[b"abc", b"123"],
                        url=f"{TEST_BASE_URL}/api/v1/results/t-1/logs",
                    )
                ),
            ),
        )

        output_path = tmp_path / "download.bin"
        await client._download("/results/t-1/logs", output_path)
        assert output_path.read_bytes() == b"abc123"

    def test_download_reads_stream_error_body_before_raising(
        self, tmp_path: Path
    ) -> None:
        client = BaseClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(
                Any,
                SyncHTTP(
                    SyncResponse(
                        status_code=404,
                        chunks=[b'{"detail":"missing result"}'],
                        json_body={"detail": "missing result"},
                        url=f"{TEST_BASE_URL}/api/v1/results/t-1/logs",
                    )
                ),
            ),
        )

        with pytest.raises(NotFoundError, match="missing result"):
            client._download("/results/t-1/logs", tmp_path / "out.bin")

    @pytest.mark.anyio
    async def test_async_download_reads_stream_error_body_before_raising(
        self, tmp_path: Path
    ) -> None:
        client = BaseAsyncClient(
            base_url=TEST_BASE_URL,
            api_key=None,
            timeout=5.0,
            http_client=cast(
                Any,
                AsyncHTTP(
                    AsyncResponse(
                        status_code=404,
                        chunks=[b'{"detail":"missing result"}'],
                        json_body={"detail": "missing result"},
                        url=f"{TEST_BASE_URL}/api/v1/results/t-1/logs",
                    )
                ),
            ),
        )

        with pytest.raises(NotFoundError, match="missing result"):
            await client._download("/results/t-1/logs", tmp_path / "out.bin")
