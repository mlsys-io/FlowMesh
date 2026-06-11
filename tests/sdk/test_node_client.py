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


class _CapturingHTTP:
    """Stub httpx.Client that records requests and returns 204 No Content."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **_kwargs: Any) -> httpx.Response:
        self.calls.append((method, url))
        return httpx.Response(204, request=httpx.Request(method, url))

    def close(self) -> None:
        return None


def _client_capturing() -> tuple[NodeClient, _CapturingHTTP]:
    client = NodeClient(base_url="http://localhost:8000", token="t")
    client._http.close()
    http = _CapturingHTTP()
    client._http = http  # type: ignore[assignment]
    return client, http


def test_destroy_all_workers_issues_delete_against_stack_workers() -> None:
    client, http = _client_capturing()
    assert client.destroy_all_workers() is True
    assert http.calls == [("DELETE", "/api/v1/stack/workers")]
