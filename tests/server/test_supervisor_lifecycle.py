"""Tests for `server.supervisor.services.lifecycle.Lifecycle`."""

import logging
from typing import Any

import pytest

from server.supervisor.services import lifecycle as lifecycle_module
from server.supervisor.services.lifecycle import Lifecycle
from shared.schemas.node import NodeInfo
from tests.server.supervisor_helpers import StubLifecycle, StubRegistry


def _build_lifecycle(base_url: str = "http://root:8000") -> Lifecycle:
    """Construct a Lifecycle skipping `__init__` — `_register_http` only needs
    `_base_url`, `_node_info`, and `logger`."""
    instance = Lifecycle.__new__(Lifecycle)
    instance._base_url = base_url
    instance._node_info = NodeInfo(
        namespace="ns",
        cluster="cl",
        alias="worker-1",
        version="0.1.0",
        started_at="2026-05-13T00:00:00Z",
        tags=[],
        last_seen="2026-05-13T00:00:00Z",
        max_gpu_count=0,
    )
    instance.logger = logging.getLogger("test.lifecycle")
    instance._on_reregister = None
    return instance


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_register_http_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _StubResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        return _StubResponse({"node_id": "nde-test"})

    monkeypatch.setenv("FLOWMESH_API_KEY", "secret-token")
    monkeypatch.setattr(lifecycle_module.httpx, "post", fake_post)

    node_id = _build_lifecycle()._register_http()

    assert node_id == "nde-test"
    assert captured["url"] == "http://root:8000/api/v1/nodes/register"
    assert captured["headers"] == {"Authorization": "Bearer secret-token"}


def test_register_http_omits_header_when_api_key_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _StubResponse:
        captured["headers"] = kwargs.get("headers")
        return _StubResponse({"node_id": "nde-test"})

    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    monkeypatch.setattr(lifecycle_module.httpx, "post", fake_post)

    _build_lifecycle()._register_http()

    assert captured["headers"] == {}


def test_reregister_if_lost_noops_when_present() -> None:
    registry = StubRegistry(exists=True)
    instance = StubLifecycle(registry, "nde-1")
    calls: list[str] = []
    instance.set_reregister_callback(calls.append)

    instance._reregister_if_lost()

    assert registry.exists_calls == ["nde-1"]
    assert instance._node_id == "nde-1"
    assert instance.published_events == []
    assert calls == []  # callback not fired when the record is present


def test_reregister_if_lost_reregisters_when_missing() -> None:
    registry = StubRegistry(exists=False)
    instance = StubLifecycle(registry, "nde-1")
    calls: list[str] = []
    instance.set_reregister_callback(calls.append)

    instance._reregister_if_lost()

    assert registry.exists_calls == ["nde-1"]
    assert instance._node_id == "nde-2"
    assert instance._unregister_published is False
    assert instance.published_events == ["SV_REGISTER"]
    assert calls == ["nde-2"]  # callback fired with the new id


def test_reregister_if_lost_swallows_callback_error() -> None:
    instance = StubLifecycle(StubRegistry(exists=False), "nde-1")

    def _boom(_new_id: str) -> None:
        raise RuntimeError("rebind failed")

    instance.set_reregister_callback(_boom)

    # a failing callback must be logged, not crash the heartbeat loop
    instance._reregister_if_lost()
    assert instance._node_id == "nde-2"
