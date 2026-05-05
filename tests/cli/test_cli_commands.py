"""CLI command behavior tests using CliRunner with mocked SDK client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from flowmesh.models import (
    WorkflowSubmitResponse,
)
from flowmesh_cli.cli import build_cli_app
from typer.testing import CliRunner

runner = CliRunner()


def _app() -> typer.Typer:
    return build_cli_app()


def _mock_client(**resource_overrides: MagicMock) -> MagicMock:
    """Build a mock FlowMesh client with resource namespaces."""
    client = MagicMock()
    for attr in (
        "workflows",
        "tasks",
        "results",
        "workers",
        "servers",
        "system",
    ):
        if attr not in resource_overrides:
            setattr(client, attr, MagicMock())
    for attr, mock in resource_overrides.items():
        setattr(client, attr, mock)
    return client


# ------------------------------------------------------------------ #
# Workflow commands
# ------------------------------------------------------------------ #


class TestWorkflowSubmit:
    def test_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(
            _app(), ["workflow", "submit", str(tmp_path / "missing.yaml")]
        )
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or result.exit_code == 1

    @patch("flowmesh_cli.commands.workflow.FlowMesh")
    def test_format_flag(self, mock_hc: MagicMock, tmp_path: Path) -> None:
        tmpl = tmp_path / "wf.json"
        tmpl.write_text('{"nodes": []}')
        client = _mock_client()
        client.workflows.submit.return_value = WorkflowSubmitResponse(
            ok=True, workflow_id="wf-1", count=1, tasks=[]
        )
        mock_hc.return_value = client

        result = runner.invoke(
            _app(), ["workflow", "submit", str(tmpl), "--format", "n8n"]
        )
        assert result.exit_code == 0
        client.workflows.submit.assert_called_once()
        call_args = client.workflows.submit.call_args
        assert call_args.args[0] == '{"nodes": []}'
        assert call_args.kwargs["workflow_format"] == "n8n"


class TestWorkflowList:
    @patch("flowmesh_cli.commands.workflow.FlowMesh")
    def test_status_filter(self, mock_hc: MagicMock) -> None:
        client = _mock_client()
        client.workflows.list.return_value = []
        mock_hc.return_value = client

        result = runner.invoke(_app(), ["workflow", "list", "--status", "DONE"])
        assert result.exit_code == 0
        client.workflows.list.assert_called_once()
        call_kwargs = client.workflows.list.call_args.kwargs
        assert call_kwargs["status"] == ["DONE"]

    @patch("flowmesh_cli.commands.workflow.FlowMesh")
    def test_query_filter_forwarded(self, mock_hc: MagicMock) -> None:
        client = _mock_client()
        client.workflows.list.return_value = []
        mock_hc.return_value = client

        result = runner.invoke(
            _app(),
            ["workflow", "list", "--query", "owner_id=u-1", "--query", "status=DONE"],
        )
        assert result.exit_code == 0
        call_kwargs = client.workflows.list.call_args.kwargs
        assert call_kwargs["query_params"] == [
            ("owner_id", "u-1"),
            ("status", "DONE"),
        ]


# ------------------------------------------------------------------ #
# Task commands
# ------------------------------------------------------------------ #


class TestTaskInfo:
    @patch("flowmesh_cli.commands.task.FlowMesh")
    def test_id_forwarded(self, mock_hc: MagicMock) -> None:
        client = _mock_client()
        task_mock = MagicMock()
        task_mock.model_dump_json.return_value = '{"task_id": "t-123"}'
        client.tasks.retrieve.return_value = task_mock
        mock_hc.return_value = client

        result = runner.invoke(_app(), ["task", "info", "t-123"])
        assert result.exit_code == 0
        client.tasks.retrieve.assert_called_once_with("t-123")


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #


class TestConfig:
    def test_config_from_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'base_url = "http://file.example:8000"\napi_key = "abc123"\n'
        )

        result = runner.invoke(
            _app(), ["config", "--source", "file", "--config", config_path.as_posix()]
        )
        assert result.exit_code == 0
        assert '"base_url": "http://file.example:8000"' in result.stdout
        assert '"api_key": "******"' in result.stdout

    def test_config_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FLOWMESH_BASE_URL": "http://env.example:8000",
                "FLOWMESH_API_KEY": "supersecret",
            },
            clear=False,
        ):
            result = runner.invoke(_app(), ["config", "--source", "env"])

        assert result.exit_code == 0
        assert '"base_url": "http://env.example:8000"' in result.stdout
        assert '"api_key": "supe***cret"' in result.stdout

    def test_config_auto_defaults_when_not_strict(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.toml"
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(
                _app(), ["config", "--source", "auto", "--config", missing.as_posix()]
            )

        assert result.exit_code == 0
        assert '"base_url": "http://localhost:8000"' in result.stdout

    def test_config_auto_fails_on_invalid_config(self, tmp_path: Path) -> None:
        invalid = tmp_path / "invalid.toml"
        invalid.write_text("base_url = 12345\n")  # Invalid type for base_url
        result = runner.invoke(
            _app(), ["config", "--source", "auto", "--config", invalid.as_posix()]
        )
        assert result.exit_code != 0
        assert "Invalid type" in result.stderr
