"""Tests for ``flowmesh_stack.node_client.NodeClient``."""

from collections.abc import Callable

import httpx
import pytest
from flowmesh.exceptions import FlowMeshConnectionError
from flowmesh_stack.node_client import NodeClient


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> NodeClient:
    """A NodeClient whose HTTP layer is backed by an ``httpx.MockTransport``."""
    client = NodeClient(base_url="http://localhost:8000", token="t")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://localhost:8000", transport=httpx.MockTransport(handler)
    )
    return client


def _raise_connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_destroy_all_workers_raises_connection_error_by_default() -> None:
    client = _mock_client(_raise_connect_error)
    with pytest.raises(FlowMeshConnectionError):
        client.destroy_all_workers()


def test_destroy_all_workers_returns_false_when_ignored() -> None:
    client = _mock_client(_raise_connect_error)
    assert client.destroy_all_workers(ignore_unreachable=True) is False


def test_destroy_all_workers_issues_delete_against_stack_workers() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(204)

    client = _mock_client(handler)
    assert client.destroy_all_workers() is True
    assert calls == [("DELETE", "/api/v1/stack/workers")]
