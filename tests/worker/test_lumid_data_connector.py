"""Tests for the lumid-data-app connector and DataRetrievalExecutor lumid branch."""

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from worker.connectors.base_connector import ConnectorError
from worker.connectors.lumid_data_connector import LumidDataConnector
from worker.executors.base_executor import ExecutionError
from worker.executors.data_retrieval_executor import DataRetrievalExecutor

from .factories import DEFAULT_WORKER_CONFIG


def _make_executor() -> DataRetrievalExecutor:
    return DataRetrievalExecutor(config=DEFAULT_WORKER_CONFIG)


_RETRIEVAL_RESULT: dict[str, Any] = {
    "run_id": "run-abc",
    "materialized_uri": "/blobs/retrievals/run-abc/result.jsonl",
    "signed_url": "http://lumid/blobs/retrievals/run-abc/result.jsonl?sig=x",
    "output_format": "jsonl",
    "access_chain": [
        {
            "op": "sql",
            "query": "SELECT symbol FROM demo.fact_ohlc_10m LIMIT 5",
            "bucket": "demo",
            "key": "fact_ohlc_10m",
            "rows_or_bytes": 5,
            "ms": 12,
        }
    ],
    "rowcount": 5,
    "size_bytes": 200,
    "tokens_in": 120,
    "tokens_out": 60,
    "steps_taken": 2,
    "replay_latency_ms": 15,
    "transcript_url": "http://lumid/v1/admin/runs/run-abc/transcript",
}

_BLOB_BYTES = b'{"symbol":"NVDA","close":900.0}\n' * 5


def _sse_body(frames: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames)


class TestLumidDataConnectorSql:
    @respx.mock
    def test_sql_round_trip_writes_file_and_returns_metadata(
        self, tmp_path: Path
    ) -> None:
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=_RETRIEVAL_RESULT)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "SELECT symbol FROM demo.fact_ohlc_10m LIMIT 5",
                mode="sql",
                out_path=out_path,
                output_format="jsonl",
            )

        assert result["success"] is True
        assert result["data"]["rowcount"] == 5
        assert (
            result["data"]["materialized_uri"]
            == "/blobs/retrievals/run-abc/result.jsonl"
        )
        assert out_path.exists()
        assert out_path.read_bytes() == _BLOB_BYTES

    @respx.mock
    def test_sql_non_2xx_returns_failure(self, tmp_path: Path) -> None:
        respx.post("/retrieve").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token="bad") as conn:
            result = conn.execute("SELECT 1", mode="sql", out_path=out_path)

        assert result["success"] is False
        error = result["error"]
        assert error is not None and "401" in error.lower()

    def test_sql_rejects_multi_element_list(self, tmp_path: Path) -> None:
        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                ["SELECT 1", "SELECT 2"], mode="sql", out_path=tmp_path / "r.jsonl"
            )
        assert result["success"] is False
        error = result["error"]
        assert error is not None and "single query" in error.lower()

    def test_unconnected_raises(self, tmp_path: Path) -> None:
        conn = LumidDataConnector(base_url="http://lumid-data")
        with pytest.raises(ConnectorError, match="not connected"):
            conn.execute("SELECT 1", mode="sql", out_path=tmp_path / "r.jsonl")


class TestLumidDataConnectorAgent:
    @respx.mock
    def test_agent_round_trip_writes_file_and_returns_metadata(
        self, tmp_path: Path
    ) -> None:
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "tool_call", "name": "sql"},
            {"type": "tool_result", "rows": 5},
            {"type": "done", "message": {}, "result": _RETRIEVAL_RESULT},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "Fetch the last 5 OHLC rows for NVDA",
                mode="agent",
                out_path=out_path,
                output_format="jsonl",
                schema_scope="demo",
            )

        assert result["success"] is True
        assert result["data"]["run_id"] == "run-abc"
        assert result["data"]["transcript_url"].endswith("transcript")
        assert result["data"]["tokens_in"] == 120
        assert result["data"]["steps_taken"] == 2
        assert out_path.exists()
        assert out_path.read_bytes() == _BLOB_BYTES

    @respx.mock
    def test_agent_error_frame_returns_failure(self, tmp_path: Path) -> None:
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "error", "error": "schema not found"},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )
        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "Fetch data",
                mode="agent",
                out_path=out_path,
            )

        assert result["success"] is False
        error = result["error"]
        assert error is not None and "schema not found" in error.lower()

    @respx.mock
    def test_agent_401_returns_failure(self, tmp_path: Path) -> None:
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with LumidDataConnector(base_url="http://lumid-data", token="bad") as conn:
            result = conn.execute(
                "Fetch data",
                mode="agent",
                out_path=tmp_path / "r.jsonl",
            )

        assert result["success"] is False
        error = result["error"]
        assert error is not None and "401" in error.lower()

    def test_agent_rejects_multi_element_list(self, tmp_path: Path) -> None:
        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                ["desc a", "desc b"],
                mode="agent",
                out_path=tmp_path / "r.jsonl",
            )
        assert result["success"] is False
        error = result["error"]
        assert error is not None and "single description" in error.lower()


class TestLumidDataConnectorS3:
    @respx.mock
    def test_s3_text_decoded_and_keyed(self, tmp_path: Path) -> None:
        respx.get("/blobs/demo/news/doc1.txt").mock(
            return_value=httpx.Response(200, text="hello world")
        )

        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "demo/news/doc1.txt",
                mode="s3",
                encoding="utf-8",
            )

        assert result["success"] is True
        assert result["data"]["demo/news/doc1.txt"] == "hello world"

    @respx.mock
    def test_s3_csv_decoded_as_dataframe_when_flag_set(self, tmp_path: Path) -> None:
        csv_bytes = b"a,b\n1,x\n2,y\n"
        respx.get("/blobs/demo/data.csv").mock(
            return_value=httpx.Response(
                200,
                content=csv_bytes,
                headers={"content-type": "text/csv"},
            )
        )

        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "demo/data.csv",
                mode="s3",
                as_dataframe=True,
            )

        assert result["success"] is True
        df = result["data"]["demo/data.csv"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @respx.mock
    def test_s3_multiple_keys_returned_as_dict(self, tmp_path: Path) -> None:
        respx.get("/blobs/a.txt").mock(return_value=httpx.Response(200, text="aaa"))
        respx.get("/blobs/b.txt").mock(return_value=httpx.Response(200, text="bbb"))

        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                ["a.txt", "b.txt"],
                mode="s3",
            )

        assert result["success"] is True
        assert result["data"]["a.txt"] == "aaa"
        assert result["data"]["b.txt"] == "bbb"
        assert result["metadata"]["file_count"] == 2


class TestDataRetrievalExecutorLumidBranch:
    @respx.mock
    def test_sql_mode_renders_template_and_yields_table_item(
        self, tmp_path: Path
    ) -> None:
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=_RETRIEVAL_RESULT)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        executor = _make_executor()
        data_cfg: dict[str, Any] = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "SELECT symbol, close FROM demo.fact_ohlc_10m LIMIT 5",
            "params": [],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)

        assert result.type == "lumid"
        assert result.count == 1
        item = result.items[0]
        assert "table" in item
        assert item["run_id"] == "run-abc"
        assert item["rows"] == 5

    @respx.mock
    def test_agent_mode_renders_description_and_yields_table_item(
        self, tmp_path: Path
    ) -> None:
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "done", "message": {}, "result": _RETRIEVAL_RESULT},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "agent",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "description": "Fetch OHLC rows for {sym}",
            "schema_scope": "demo",
            "params": [{"label": "sym", "data": {"type": "list", "items": ["NVDA"]}}],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)

        assert result.type == "lumid"
        assert result.count == 1
        item = result.items[0]
        assert "NVDA" in item["description"]
        assert item["run_id"] == "run-abc"
        assert item["transcript_url"].endswith("transcript")
        assert "table" in item

    @respx.mock
    def test_s3_mode_serializes_content(self, tmp_path: Path) -> None:
        respx.get("/blobs/demo/unstructured/news-html/doc1.html").mock(
            return_value=httpx.Response(200, text="<html>news</html>")
        )

        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "s3",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "demo/unstructured/news-html/doc1.html",
            "params": [],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)

        assert result.type == "lumid"
        assert result.count == 1
        assert result.items[0]["keys"] == ["demo/unstructured/news-html/doc1.html"]

    def test_bad_mode_raises_execution_error(self, tmp_path: Path) -> None:
        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "unknown",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with pytest.raises(ExecutionError, match="mode"):
            executor._run_lumid(data_cfg, {}, out_dir)

    def test_bad_output_format_raises_execution_error(self, tmp_path: Path) -> None:
        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "SELECT 1",
            "output_format": "raw",
            "params": [],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with pytest.raises(ExecutionError, match="output_format"):
            executor._run_lumid(data_cfg, {}, out_dir)

    def test_missing_token_raises(self, tmp_path: Path) -> None:
        """lumid_data_token is required on every lumid node."""
        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "template": "SELECT 1",
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with pytest.raises(KeyError, match="lumid_data_token"):
            executor._run_lumid(data_cfg, {}, out_dir)

    @respx.mock
    def test_agent_mode_materialized_uri_from_data(self, tmp_path: Path) -> None:
        """materialized_uri in the item comes from outcome['data'], not metadata."""
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "done", "message": {}, "result": _RETRIEVAL_RESULT},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        executor = _make_executor()
        data_cfg = {
            "type": "lumid",
            "mode": "agent",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "description": "Fetch OHLC rows",
            "params": [],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)

        item = result.items[0]
        assert item["materialized_uri"] == _RETRIEVAL_RESULT["materialized_uri"]

    @respx.mock
    def test_sql_mode_access_chain_round_trip(self, tmp_path: Path) -> None:
        """access_chain in the item equals the mocked RetrievalResult access_chain."""
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=_RETRIEVAL_RESULT)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        executor = _make_executor()
        data_cfg: dict[str, Any] = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "SELECT symbol FROM demo.fact_ohlc_10m LIMIT 5",
            "params": [],
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)

        item = result.items[0]
        assert item["access_chain"] == _RETRIEVAL_RESULT["access_chain"]
        assert result.items[0]["access_chain"][0]["op"] == "sql"

    @respx.mock
    def test_sql_mode_zero_rows_raises_when_required(self, tmp_path: Path) -> None:
        """require_non_empty_rows=true raises ExecutionError when rowcount is 0."""
        empty_result = {**_RETRIEVAL_RESULT, "rowcount": 0, "size_bytes": 0}
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=empty_result)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=b"")
        )

        executor = _make_executor()
        data_cfg: dict[str, Any] = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "SELECT symbol FROM demo.fact_ohlc_10m LIMIT 0",
            "params": [],
            "require_non_empty_rows": True,
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(ExecutionError, match="zero rows"):
            executor._run_lumid(data_cfg, {}, out_dir)

    @respx.mock
    def test_sql_mode_zero_rows_ok_when_not_required(self, tmp_path: Path) -> None:
        """require_non_empty_rows=false does not raise for an empty result."""
        empty_result = {**_RETRIEVAL_RESULT, "rowcount": 0, "size_bytes": 0}
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=empty_result)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=b"")
        )

        executor = _make_executor()
        data_cfg: dict[str, Any] = {
            "type": "lumid",
            "mode": "sql",
            "lumid_data_url": "http://lumid-data",
            "lumid_data_token": "tok",
            "template": "SELECT symbol FROM demo.fact_ohlc_10m LIMIT 0",
            "params": [],
            "require_non_empty_rows": False,
        }
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = executor._run_lumid(data_cfg, {}, out_dir)
        assert result.count == 1
        assert result.items[0]["rows"] == 0


class TestLumidDataConnectorAgentSseEdgeCases:
    @respx.mock
    def test_no_done_frame_returns_failure(self, tmp_path: Path) -> None:
        """SSE stream with only iteration/tool frames (no done) yields success=False."""
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "tool_call", "name": "sql"},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )

        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "Fetch data",
                mode="agent",
                out_path=tmp_path / "r.jsonl",
            )

        assert result["success"] is False
        error = result["error"]
        assert error is not None and "done" in error.lower()

    @respx.mock
    def test_done_frame_with_null_result_returns_failure(self, tmp_path: Path) -> None:
        """SSE done frame present but result is null yields success=False."""
        frames: list[dict[str, Any]] = [
            {"type": "iteration", "step": 1},
            {"type": "done", "message": {}, "result": None},
        ]
        respx.post("/agent/v1").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(frames),
                headers={"content-type": "text/event-stream"},
            )
        )

        with LumidDataConnector(base_url="http://lumid-data", token="tok") as conn:
            result = conn.execute(
                "Fetch data",
                mode="agent",
                out_path=tmp_path / "r.jsonl",
            )

        assert result["success"] is False
        error = result["error"]
        assert error is not None and "result" in error.lower()


class TestLumidDataConnectorTokenFromEnv:
    @respx.mock
    def test_env_token_ignored_when_token_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LUMID_DATA_TOKEN env var is not read; token=None omits Authorization."""
        monkeypatch.setenv("LUMID_DATA_TOKEN", "env-secret-token")

        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=_RETRIEVAL_RESULT)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token=None) as conn:
            result = conn.execute(
                "SELECT 1",
                mode="sql",
                out_path=out_path,
                output_format="jsonl",
            )

        assert result["success"] is True
        sent_request = respx.calls[0].request
        assert "authorization" not in sent_request.headers

    @respx.mock
    def test_explicit_token_sent_as_bearer(self, tmp_path: Path) -> None:
        """A token passed explicitly to the ctor is sent as Authorization: Bearer."""
        respx.post("/retrieve").mock(
            return_value=httpx.Response(200, json=_RETRIEVAL_RESULT)
        )
        respx.get("/blobs/retrievals/run-abc/result.jsonl").mock(
            return_value=httpx.Response(200, content=_BLOB_BYTES)
        )

        out_path = tmp_path / "result.jsonl"
        with LumidDataConnector(base_url="http://lumid-data", token="my-pat") as conn:
            result = conn.execute(
                "SELECT 1",
                mode="sql",
                out_path=out_path,
                output_format="jsonl",
            )

        assert result["success"] is True
        sent_request = respx.calls[0].request
        assert sent_request.headers["authorization"] == "Bearer my-pat"
