"""Tests for `get_http_destination` covering both explicit
spec.output.destination and the WORKER_UPLOAD_RESULTS opt-in.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from worker.executors.utils.checkpoints import (
    HTTPDestination,
    get_http_destination,
)


def _spec(destination: dict[str, Any] | None) -> Any:
    """Build a minimal spec mock with .output.destination as `destination`."""
    spec = MagicMock()
    if destination is None:
        spec.output = None
    else:
        dest = MagicMock(
            type=destination.get("type", "http"),
            method=destination.get("method"),
            url=destination.get("url"),
            headers=destination.get("headers"),
            timeoutSec=destination.get("timeoutSec"),
        )
        spec.output = MagicMock(destination=dest)
    return spec


# ---- explicit spec.output.destination ----------------------------------


def test_explicit_destination_returned_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    monkeypatch.delenv("FLOWMESH_BASE_URL", raising=False)
    spec = _spec({"type": "http", "url": "http://host:8000/api/v1/results"})
    out = get_http_destination(spec)
    assert isinstance(out, HTTPDestination)
    assert out.method == "POST"
    assert out.url == "http://host:8000/api/v1/results"
    assert out.headers == {}
    assert out.timeout == 30
    assert out.ignore_error is False


def test_explicit_destination_url_falls_back_to_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://host:8000")
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    spec = _spec({"type": "http"})
    out = get_http_destination(spec)
    assert out is not None
    assert out.url == "http://host:8000/api/v1/results"
    assert out.ignore_error is False


def test_explicit_destination_honors_method_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    spec = _spec(
        {
            "type": "http",
            "url": "http://x/y",
            "method": "PUT",
            "headers": {"X-Custom": "v"},
            "timeoutSec": 5,
        }
    )
    out = get_http_destination(spec)
    assert out is not None
    assert out.method == "PUT"
    assert out.timeout == 5
    assert out.headers == {"X-Custom": "v"}


def test_destination_type_other_than_http_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKER_UPLOAD_RESULTS", raising=False)
    spec = _spec({"type": "s3", "url": "s3://bucket/key"})
    assert get_http_destination(spec) is None


# ---- WORKER_UPLOAD_RESULTS opt-in -------------------------------------


def test_upload_results_no_destination_no_env_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKER_UPLOAD_RESULTS", raising=False)
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://host:8000")
    spec = _spec(None)
    assert get_http_destination(spec) is None


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "True", "yes", "on"])
def test_upload_results_enabled_via_env(
    truthy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", truthy)
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://host:8000")
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    spec = _spec(None)
    out = get_http_destination(spec)
    assert out is not None
    assert out.ignore_error is True


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "", "  "])
def test_upload_results_disabled_via_env(
    falsy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", falsy)
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://host:8000")
    spec = _spec(None)
    assert get_http_destination(spec) is None


def test_upload_results_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "true")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh-host:8000/")
    monkeypatch.delenv("FLOWMESH_API_KEY", raising=False)
    spec = _spec(None)
    out = get_http_destination(spec)
    assert out is not None
    assert out.url == "http://flowmesh-host:8000/api/v1/results"
    assert out.ignore_error is True
    assert out.method == "POST"


def test_upload_results_without_base_url_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload enabled but no FLOWMESH_BASE_URL — nowhere to send."""
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "true")
    monkeypatch.delenv("FLOWMESH_BASE_URL", raising=False)
    spec = _spec(None)
    assert get_http_destination(spec) is None


def test_explicit_destination_wins_over_upload_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit spec.output.destination is honored even when WORKER_UPLOAD_RESULTS
    is on."""
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "true")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh-host:8000")
    spec = _spec({"type": "http", "url": "http://other:9999/sink"})
    out = get_http_destination(spec)
    assert out is not None
    assert out.url == "http://other:9999/sink"
    # Explicit path is hard-fail; runner won't soft-fail on host blips.
    assert out.ignore_error is False


# ---- Authorization header ---------------------------------------------


def test_api_key_added_for_flowmesh_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh.example")
    spec = _spec({"type": "http", "url": "http://flowmesh.example/api/v1/results"})
    out = get_http_destination(spec)
    assert out is not None
    assert out.headers["Authorization"] == "Bearer flm-test"


def test_api_key_not_added_for_external_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh.example")
    spec = _spec({"type": "http", "url": "https://external.example/results"})
    out = get_http_destination(spec)
    assert out is not None
    assert "Authorization" not in out.headers


def test_existing_authorization_header_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
    spec = _spec(
        {
            "type": "http",
            "url": "http://x/y",
            "headers": {"Authorization": "Bearer caller-supplied"},
        }
    )
    out = get_http_destination(spec)
    assert out is not None
    assert out.headers["Authorization"] == "Bearer caller-supplied"
