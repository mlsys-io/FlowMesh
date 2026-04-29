"""Tests for ``flowmesh_stack.node_client.NodeClient``."""

from typing import Any

import httpx
import pytest
from flowmesh.exceptions import FlowMeshConnectionError
from flowmesh_stack.node_client import NodeClient


class _RaisingHTTP:
    """Stub httpx.Client that raises ``ConnectError`` on every request."""

    def request(self, method: str, url: str, **_kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError(
            "connection refused",
            request=httpx.Request(method, url),
        )

    def close(self) -> None:
        return None


def _client_against_unreachable() -> NodeClient:
    client = NodeClient(base_url="http://localhost:1", token="t")
    client._http.close()
    client._http = _RaisingHTTP()  # type: ignore[assignment]
    return client


def test_destroy_all_workers_raises_connection_error_by_default() -> None:
    client = _client_against_unreachable()
    with pytest.raises(FlowMeshConnectionError):
        client.destroy_all_workers()


def test_destroy_all_workers_returns_false_when_ignored() -> None:
    client = _client_against_unreachable()
    assert client.destroy_all_workers(ignore_unreachable=True) is False
