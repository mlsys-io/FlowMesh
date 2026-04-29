"""Tests for ``flowmesh_stack.node_client.NodeClient``."""

import logging
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


def test_destroy_all_workers_returns_false_and_warns_when_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client_against_unreachable()
    with caplog.at_level(logging.WARNING, logger="flowmesh_stack.node_client"):
        result = client.destroy_all_workers(ignore_unreachable=True)
    assert result is False
    assert any("destroy_all_workers" in r.message for r in caplog.records)


def test_destroy_worker_raises_connection_error_by_default() -> None:
    client = _client_against_unreachable()
    with pytest.raises(FlowMeshConnectionError):
        client.destroy_worker("worker-1")


def test_destroy_worker_returns_false_and_warns_when_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client_against_unreachable()
    with caplog.at_level(logging.WARNING, logger="flowmesh_stack.node_client"):
        result = client.destroy_worker("worker-1", ignore_unreachable=True)
    assert result is False
    assert any("destroy_worker(worker-1)" in r.message for r in caplog.records)


def test_stop_worker_raises_connection_error_by_default() -> None:
    client = _client_against_unreachable()
    with pytest.raises(FlowMeshConnectionError):
        client.stop_worker("worker-1")


def test_stop_worker_returns_false_and_warns_when_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client_against_unreachable()
    with caplog.at_level(logging.WARNING, logger="flowmesh_stack.node_client"):
        result = client.stop_worker("worker-1", ignore_unreachable=True)
    assert result is False
    assert any("stop_worker(worker-1)" in r.message for r in caplog.records)
