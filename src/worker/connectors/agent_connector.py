"""Agent connector — calls lumid.data ``/retrieve/v1`` for NL-driven retrieval."""

from pathlib import Path
from typing import Any

from lumid_data.sdk import Client as LumidDataClient

from .base_connector import BaseConnector, ConnectorError


class AgentConnector(BaseConnector):
    """Wraps :meth:`lumid_data.sdk.Client.retrieve_to_file`."""

    name = "agent_connector"

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        verify: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url
        self._token = token
        self._verify = verify
        self._client: LumidDataClient | None = None

    def connect(self) -> None:
        self._client = LumidDataClient(base_url=self._base_url, token=self._token)

    def execute(
        self,
        query: str | list[str],
        *,
        schema_scope: str | None = None,
        out_path: Path,
        output_format: str = "jsonl",
        max_steps: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run a single agent retrieval against lumid.data."""
        if isinstance(query, list):
            if len(query) != 1:
                raise ConnectorError(
                    "agent connector accepts a single description; "
                    f"got {len(query)} entries"
                )
            description = query[0]
        else:
            description = query
        if self._client is None:
            raise ConnectorError("agent connector is not connected")
        try:
            result = self._client.retrieve_to_file(
                description,
                schema_scope=schema_scope,
                out_path=out_path,
                output_format=output_format,
                max_steps=max_steps,
                model=model,
                verify=self._verify,
            )
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"agent retrieval failed: {exc}",
                "metadata": {},
            }
        return {
            "success": True,
            "data": {
                "out_path": out_path.as_posix(),
                "rowcount": result.rowcount,
                "size_bytes": result.size_bytes,
                "access_chain": [step.model_dump() for step in result.access_chain],
                "run_id": result.run_id,
                "transcript_url": result.transcript_url,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "steps_taken": result.steps_taken,
                "replay_latency_ms": result.replay_latency_ms,
            },
            "error": None,
            "metadata": {
                "materialized_uri": result.materialized_uri,
                "output_format": result.output_format,
            },
        }
