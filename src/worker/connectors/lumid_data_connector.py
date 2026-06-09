"""HTTP connector for lumid-data-app.

Supports SQL queries, agent-driven retrieval, and S3-style blob fetches.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd
from PIL import Image

from .base_connector import BaseConnector, ConnectorError, ConnectorResult

logger = logging.getLogger(__name__)


class LumidDataConnector(BaseConnector):
    """HTTP connector for lumid-data-app.

    Supports three retrieval modes dispatched via :meth:`execute`:
    - ``sql``: run a SQL query and materialise the result to a local file.
    - ``agent``: drive the NL data agent via SSE and materialise the result.
    - ``s3``: fetch raw blobs by key and decode them by content type.
    """

    name = "lumid_data"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        verify: bool = True,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        """Initialise the connector.

        Args:
            base_url: Base URL of the lumid-data-app instance.
            token: Caller-supplied bearer token (the user's PAT, passed from
                the workflow spec's ``lumid_data_token`` field).  When ``None``
                no ``Authorization`` header is sent.
            verify: Whether to verify TLS certificates. Defaults to ``True`` so
                bearer tokens are never sent over an unverified TLS channel.
            timeout: Request timeout in seconds.
        """
        super().__init__(**kwargs)
        self._base_url = base_url
        self._token = token
        self._verify = verify
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def connect(self) -> None:
        """Open the underlying HTTP client."""
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            verify=self._verify,
            headers=headers,
        )

    def disconnect(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def execute(
        self,
        query: str | list[str],
        mode: str,
        out_path: Path | None = None,
        output_format: str = "jsonl",
        schema_scope: str | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        encoding: str = "utf-8",
        as_dataframe: bool = False,
    ) -> ConnectorResult:
        """Dispatch a retrieval request in one of three modes.

        Args:
            query: SQL string (sql mode), NL description (agent mode), or key /
                list of keys (s3 mode).
            mode: One of ``"sql"``, ``"agent"``, or ``"s3"``.
            out_path: Where to write the materialised result file (sql / agent).
            output_format: ``"jsonl"`` or ``"csv"`` (sql / agent).
            schema_scope: Optional schema restriction directive (agent only).
            max_steps: Maximum agent iterations (agent only).
            model: Model override (agent only).
            encoding: Text encoding for blob decode (s3 only).
            as_dataframe: Decode CSV blobs as DataFrames (s3 only).

        Returns:
            A :class:`ConnectorResult` envelope.
        """
        if self._client is None:
            raise ConnectorError("lumid_data connector is not connected")
        try:
            if mode == "sql":
                return self._execute_sql(
                    query, out_path=out_path, output_format=output_format
                )
            if mode == "agent":
                return self._execute_agent(
                    query,
                    out_path=out_path,
                    output_format=output_format,
                    schema_scope=schema_scope,
                    max_steps=max_steps,
                    model=model,
                )
            if mode == "s3":
                return self._execute_s3(
                    query, encoding=encoding, as_dataframe=as_dataframe
                )
            return {
                "success": False,
                "data": None,
                "error": f"unsupported mode: {mode!r}",
                "metadata": {},
            }
        except ConnectorError as exc:
            return {"success": False, "data": None, "error": str(exc), "metadata": {}}
        except httpx.TimeoutException as exc:
            return {
                "success": False,
                "data": None,
                "error": f"request timed out: {exc}",
                "metadata": {},
            }
        except httpx.HTTPError as exc:
            return {
                "success": False,
                "data": None,
                "error": f"HTTP error: {exc}",
                "metadata": {},
            }

    def _execute_sql(
        self,
        query: str | list[str],
        out_path: Path | None,
        output_format: str,
    ) -> ConnectorResult:
        if isinstance(query, list):
            if len(query) != 1:
                raise ConnectorError(
                    f"sql mode accepts a single query; got {len(query)} entries"
                )
            sql = query[0]
        else:
            sql = query

        if self._client is None:
            raise ConnectorError("not connected")
        resp = self._client.post(
            "/retrieve", json={"sql": sql, "output_format": output_format}
        )
        if not resp.is_success:
            return {
                "success": False,
                "data": None,
                "error": f"POST /retrieve returned {resp.status_code}: {resp.text}",
                "metadata": {},
            }
        result: dict[str, Any] = resp.json()
        materialized_uri: str = result["materialized_uri"]
        result_format: str = result["output_format"]

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with self._client.stream("GET", materialized_uri) as r:
                r.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)

        return {
            "success": True,
            "data": {
                "out_path": out_path.as_posix() if out_path is not None else None,
                "rowcount": result["rowcount"],
                "size_bytes": result["size_bytes"],
                "access_chain": result["access_chain"],
                "run_id": result["run_id"],
                "materialized_uri": materialized_uri,
                "output_format": result_format,
            },
            "error": None,
            "metadata": {
                "materialized_uri": materialized_uri,
                "output_format": result_format,
            },
        }

    def _execute_agent(
        self,
        query: str | list[str],
        out_path: Path | None,
        output_format: str,
        schema_scope: str | None,
        max_steps: int | None,
        model: str | None,
    ) -> ConnectorResult:
        if isinstance(query, list):
            if len(query) != 1:
                raise ConnectorError(
                    f"agent mode accepts a single description; got {len(query)} entries"
                )
            description = query[0]
        else:
            description = query

        if schema_scope is not None:
            content = (
                f"Restrict retrieval to schema(s): {schema_scope}.\n\n{description}"
            )
        else:
            content = description
        messages = [{"role": "user", "content": content}]

        payload: dict[str, Any] = {"messages": messages}
        if model is not None:
            payload["model"] = model
        if max_steps is not None:
            payload["max_iterations"] = max_steps

        if self._client is None:
            raise ConnectorError("not connected")
        resp = self._client.post("/agent/v1", json=payload)
        if not resp.is_success:
            return {
                "success": False,
                "data": None,
                "error": f"POST /agent/v1 returned {resp.status_code}: {resp.text}",
                "metadata": {},
            }

        body = resp.text
        frames: list[dict[str, Any]] = []
        for line in body.splitlines():
            if line.startswith("data: "):
                try:
                    frames.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        for frame in frames:
            if frame.get("type") == "error":
                return {
                    "success": False,
                    "data": None,
                    "error": frame.get("error", "agent returned an error frame"),
                    "metadata": {},
                }

        done_frame: dict[str, Any] | None = None
        for frame in reversed(frames):
            if frame.get("type") == "done":
                done_frame = frame
                break

        if done_frame is None:
            return {
                "success": False,
                "data": None,
                "error": "agent returned no done frame",
                "metadata": {},
            }

        result: dict[str, Any] | None = done_frame.get("result")
        if result is None:
            return {
                "success": False,
                "data": None,
                "error": "agent returned no retrieval result",
                "metadata": {},
            }

        materialized_uri: str = result["materialized_uri"]

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with self._client.stream("GET", materialized_uri) as r:
                r.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)

        return {
            "success": True,
            "data": {
                "out_path": out_path.as_posix() if out_path is not None else None,
                "rowcount": result["rowcount"],
                "size_bytes": result["size_bytes"],
                "access_chain": result["access_chain"],
                "run_id": result["run_id"],
                "transcript_url": result["transcript_url"],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "steps_taken": result["steps_taken"],
                "replay_latency_ms": result["replay_latency_ms"],
                "materialized_uri": materialized_uri,
            },
            "error": None,
            "metadata": {
                "materialized_uri": materialized_uri,
                "output_format": result["output_format"],
            },
        }

    def _execute_s3(
        self,
        query: str | list[str],
        encoding: str,
        as_dataframe: bool,
    ) -> ConnectorResult:
        keys: list[str] = [query] if isinstance(query, str) else query
        if self._client is None:
            raise ConnectorError("not connected")

        contents: dict[str, Any] = {}
        file_info: dict[str, dict[str, Any]] = {}
        for key in keys:
            encoded_key = quote(key, safe="/")
            resp = self._client.get(f"/blobs/{encoded_key}")
            if not resp.is_success:
                return {
                    "success": False,
                    "data": None,
                    "error": (
                        f"GET /blobs/{encoded_key} returned "
                        f"{resp.status_code}: {resp.text}"
                    ),
                    "metadata": {},
                }
            content_type = resp.headers.get("content-type", "")
            raw: bytes = resp.content
            file_info[key] = {"size": len(raw), "content_type": content_type}

            if as_dataframe and key.lower().endswith(".csv"):
                contents[key] = pd.read_csv(io.BytesIO(raw))
            elif content_type.startswith("image/"):
                contents[key] = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                try:
                    contents[key] = raw.decode(encoding)
                except UnicodeDecodeError:
                    contents[key] = raw

        return {
            "success": True,
            "data": contents,
            "error": None,
            "metadata": {
                "file_count": len(keys),
                "files": file_info,
            },
        }
