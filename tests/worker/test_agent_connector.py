"""Tests for the lumid.data agent connector and DataRetrievalExecutor branch."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from worker.connectors.agent_connector import AgentConnector
from worker.executors.base_executor import ExecutionError
from worker.executors.data_retrieval_executor import DataRetrievalExecutor

from .factories import DEFAULT_WORKER_CONFIG


def _make_executor():
    """Construct a DataRetrievalExecutor with the test-suite default config."""
    return DataRetrievalExecutor(config=DEFAULT_WORKER_CONFIG)


def _stub_retrieval_result(out_path: Path, rows: list[dict[str, Any]]) -> MagicMock:
    """Materialize ``rows`` to ``out_path`` and return a stubbed RetrievalResult."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result = MagicMock()
    result.rowcount = len(rows)
    result.size_bytes = out_path.stat().st_size
    result.access_chain = [
        MagicMock(
            model_dump=lambda: {
                "op": "sql",
                "query": "SELECT 1",
                "rows_or_bytes": len(rows),
            }
        )
    ]
    result.run_id = "run-fake"
    result.transcript_url = "http://lumid/v1/admin/runs/run-fake"
    result.tokens_in = 100
    result.tokens_out = 50
    result.steps_taken = 3
    result.replay_latency_ms = 12
    result.materialized_uri = "s3://lumid-data/retrievals/run-fake/result.jsonl"
    result.output_format = "jsonl"
    return result


class TestAgentConnector:
    def test_execute_round_trips_through_retrieve_to_file(self, tmp_path: Path):
        out = tmp_path / "result.jsonl"
        rows = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]

        with patch("worker.connectors.agent_connector.LumidDataClient") as client_cls:
            instance = client_cls.return_value
            instance.retrieve_to_file.side_effect = (
                lambda *a, **kw: _stub_retrieval_result(out, rows)
            )
            connector = AgentConnector(base_url="http://lumid", token="tok")
            connector.connect()
            outcome = connector.execute(
                "fetch all rows",
                schema_scope="schema.t",
                out_path=out,
            )

        assert outcome["success"] is True
        assert outcome["data"]["rowcount"] == 2
        assert outcome["data"]["run_id"] == "run-fake"
        assert outcome["data"]["transcript_url"].endswith("run-fake")
        assert outcome["metadata"]["materialized_uri"].startswith("s3://")
        instance.retrieve_to_file.assert_called_once()
        call_kwargs = instance.retrieve_to_file.call_args.kwargs
        assert call_kwargs["schema_scope"] == "schema.t"
        assert call_kwargs["out_path"] == out

    def test_rejects_multi_element_query_list(self, tmp_path: Path):
        with patch("worker.connectors.agent_connector.LumidDataClient"):
            connector = AgentConnector(base_url="http://lumid")
            connector.connect()
            with pytest.raises(Exception, match="single description"):
                connector.execute(
                    ["a", "b"],
                    schema_scope="schema.t",
                    out_path=tmp_path / "result.jsonl",
                )

    def test_accepts_single_element_list(self, tmp_path: Path):
        out = tmp_path / "result.jsonl"
        with patch("worker.connectors.agent_connector.LumidDataClient") as client_cls:
            instance = client_cls.return_value
            instance.retrieve_to_file.side_effect = (
                lambda *a, **kw: _stub_retrieval_result(out, [{"x": 1}])
            )
            connector = AgentConnector(base_url="http://lumid")
            connector.connect()
            outcome = connector.execute(
                ["fetch one"],
                schema_scope="schema.t",
                out_path=out,
            )
        assert outcome["success"] is True
        assert outcome["data"]["rowcount"] == 1
        # single-element list got unwrapped before passing to the SDK
        assert instance.retrieve_to_file.call_args.args[0] == "fetch one"

    def test_surfaces_sdk_failure_as_unsuccessful_outcome(self, tmp_path: Path):
        with patch("worker.connectors.agent_connector.LumidDataClient") as client_cls:
            instance = client_cls.return_value
            instance.retrieve_to_file.side_effect = RuntimeError("agent dead")
            connector = AgentConnector(base_url="http://lumid")
            connector.connect()
            outcome = connector.execute(
                "fetch",
                schema_scope="schema.t",
                out_path=tmp_path / "result.jsonl",
            )
        assert outcome["success"] is False
        assert outcome["data"] is None
        assert "agent dead" in outcome["error"]

    def test_unconnected_connector_raises(self, tmp_path: Path):
        connector = AgentConnector(base_url="http://lumid")
        with pytest.raises(Exception, match="not connected"):
            connector.execute(
                "fetch",
                schema_scope="schema.t",
                out_path=tmp_path / "result.jsonl",
            )

    def test_context_manager_connects_and_disconnects(self, tmp_path: Path):
        with patch("worker.connectors.agent_connector.LumidDataClient") as client_cls:
            instance = client_cls.return_value
            instance.retrieve_to_file.side_effect = (
                lambda *a, **kw: _stub_retrieval_result(
                    tmp_path / "result.jsonl", [{"x": 1}]
                )
            )
            with AgentConnector(base_url="http://lumid") as conn:
                outcome = conn.execute(
                    "fetch",
                    schema_scope="schema.t",
                    out_path=tmp_path / "result.jsonl",
                )
            assert outcome["success"] is True


class TestDataRetrievalExecutorAgentBranch:
    """Black-box: drive _run_agent through the executor surface with a stubbed
    LumidDataClient. Verifies the rendered description, item shape, and table
    materialization."""

    def test_renders_description_from_params_and_yields_table_item(
        self, tmp_path: Path
    ):
        rows = [{"symbol": "NVDA", "close": 100.5}]
        materialized = (
            tmp_path / "out_dir" / "artifacts" / "agent_retrievals" / "result_0.jsonl"
        )

        executor = _make_executor()
        data_cfg = {
            "type": "agent",
            "description": "fetch NVDA quotes for {ref0}",
            "schema_scope": "schema.t",
            "lumid_data_url": "http://lumid",
            "params": [
                {"label": "ref0", "data": {"type": "list", "items": ["2024-01-01"]}}
            ],
        }
        context: dict[str, Any] = {}
        out_dir = tmp_path / "out_dir"
        out_dir.mkdir()

        with patch("worker.connectors.agent_connector.LumidDataClient") as client_cls:
            instance = client_cls.return_value
            instance.retrieve_to_file.side_effect = (
                lambda *a, **kw: _stub_retrieval_result(materialized, rows)
            )
            result = executor._run_agent(data_cfg, context, out_dir)

        assert result["ok"] is True
        assert result["count"] == 1
        item = result["items"][0]
        assert "fetch NVDA quotes for 2024-01-01" in item["description"]
        assert item["rows"] == 1
        assert item["run_id"] == "run-fake"
        assert "table" in item  # serialize_dataframe output
        # Verify the SDK was called with the rendered description
        call_args = instance.retrieve_to_file.call_args
        assert "2024-01-01" in call_args.args[0]
        assert call_args.kwargs["schema_scope"] == "schema.t"

    def test_rejects_unsupported_output_format(self, tmp_path: Path):
        executor = _make_executor()
        data_cfg = {
            "type": "agent",
            "description": "fetch",
            "schema_scope": "schema.t",
            "lumid_data_url": "http://lumid",
            "output_format": "raw",
            "params": [],
        }
        out_dir = tmp_path / "out_dir"
        out_dir.mkdir()
        with pytest.raises(ExecutionError, match="output_format"):
            executor._run_agent(data_cfg, {}, out_dir)

    def test_loads_jsonl_into_dataframe(self, tmp_path: Path):

        executor = _make_executor()
        path = tmp_path / "rows.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"a": 1, "b": "x"}) + "\n")
            f.write(json.dumps({"a": 2, "b": "y"}) + "\n")
        df = executor._load_table(path, "jsonl")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_loads_csv_into_dataframe(self, tmp_path: Path):

        executor = _make_executor()
        path = tmp_path / "rows.csv"
        path.write_text("a,b\n1,x\n2,y\n")
        df = executor._load_table(path, "csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_file_returns_empty_dataframe(self, tmp_path: Path):

        executor = _make_executor()
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        df = executor._load_table(path, "jsonl")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
