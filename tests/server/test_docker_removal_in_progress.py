"""Tests for the supervisor Docker adapter's handling of a concurrent remove."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from docker.errors import APIError, NotFound
from requests import Response

from server.hooks import PrincipalContext
from server.supervisor.adapters import docker as docker_adapter
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
    _is_removal_in_progress,
)

_IN_PROGRESS = "removal of container gpu_0 is already in progress"


def _api_error(status_code: int, explanation: str | None) -> APIError:
    # APIError.status_code is a read-only property over .response.
    response = Response()
    response.status_code = status_code
    return APIError("boom", response=response, explanation=explanation)


def _adapter(docker_client: MagicMock) -> DockerWorkerAdapter:
    adapter = DockerWorkerAdapter(
        token=WorkerTokenType("worker-token"),
        alias="gpu_0",
        container_name="gpu_0",
        cuda_devices=None,
        gpu_arch=None,
        config=DockerWorkerConfig(
            worker_type=WorkerType.CPU,
            results_dir="/results",
            hf_cache_dir="/hf-cache",
            enable_ssh=False,
        ),
        docker_client=docker_client,
        owner=PrincipalContext(
            principal_id="test-user",
            org_id="test-org",
            external_id="test-user",
            principal_type="user",
            scopes=[],
        ),
    )
    adapter._hardware = {}
    return adapter


def _stale_container(remove_error: Exception | None) -> MagicMock:
    container = MagicMock(status="exited")
    container.remove.side_effect = remove_error
    return container


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_POLL", 0)


class TestIsRemovalInProgress:
    def test_matches_the_concurrent_removal_409(self) -> None:
        assert _is_removal_in_progress(_api_error(409, _IN_PROGRESS)) is True

    def test_rejects_other_409s(self) -> None:
        name_taken = _api_error(
            409, 'Conflict. The container name "/gpu_0" is already in use'
        )
        running = _api_error(
            409, "You cannot remove a running container. Stop the container"
        )
        assert _is_removal_in_progress(name_taken) is False
        assert _is_removal_in_progress(running) is False

    def test_rejects_non_409_even_with_matching_text(self) -> None:
        assert _is_removal_in_progress(_api_error(500, _IN_PROGRESS)) is False

    def test_tolerates_missing_explanation(self) -> None:
        assert _is_removal_in_progress(_api_error(409, None)) is False

    def test_rejects_non_api_errors(self) -> None:
        assert _is_removal_in_progress(RuntimeError(_IN_PROGRESS)) is False


class TestWaitContainerGone:
    def test_returns_true_once_the_container_disappears(self) -> None:
        client = MagicMock()
        client.containers.get.side_effect = [
            SimpleNamespace(),
            SimpleNamespace(),
            NotFound("gone"),
        ]
        assert _adapter(client)._wait_container_gone() is True
        assert client.containers.get.call_count == 3

    def test_times_out_when_it_never_goes(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        client = MagicMock()
        client.containers.get.return_value = SimpleNamespace()
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._wait_container_gone() is False
        assert "is still present" in caplog.text

    def test_keeps_waiting_through_a_transient_inspect_error(self) -> None:
        client = MagicMock()
        client.containers.get.side_effect = [
            RuntimeError("transport blip"),
            SimpleNamespace(),
            NotFound("gone"),
        ]
        assert _adapter(client)._wait_container_gone() is True
        assert client.containers.get.call_count == 3

    def test_timeout_reports_the_inspect_error_when_presence_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        client = MagicMock()
        client.containers.get.side_effect = RuntimeError("daemon unreachable")
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._wait_container_gone() is False
        assert "Could not confirm removal" in caplog.text
        assert "daemon unreachable" in caplog.text


class TestStartWithStaleContainer:
    def _client(self, stale: MagicMock, *after_remove: object) -> MagicMock:
        client = MagicMock()
        client.containers.get.side_effect = [stale, *after_remove]
        return client

    def test_waits_out_a_concurrent_removal_then_starts(self) -> None:
        stale = _stale_container(_api_error(409, _IN_PROGRESS))
        client = self._client(stale, SimpleNamespace(), NotFound("gone"))
        assert _adapter(client)._start() is True
        assert client.containers.run.call_args.kwargs["name"] == "gpu_0"

    def test_starts_when_the_container_is_gone_before_the_remove(self) -> None:
        client = self._client(_stale_container(NotFound("gone")))
        assert _adapter(client)._start() is True
        client.containers.run.assert_called_once()

    def test_other_conflicts_fail_the_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stale = _stale_container(
            _api_error(409, "You cannot remove a running container")
        )
        client = self._client(stale)
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._start() is False
        client.containers.run.assert_not_called()
        assert "You cannot remove a running container" in caplog.text

    def test_a_removal_that_never_completes_fails_the_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        stale = _stale_container(_api_error(409, _IN_PROGRESS))
        client = self._client(stale, SimpleNamespace())
        assert _adapter(client)._start() is False
        client.containers.run.assert_not_called()

    def test_a_running_container_is_kept(self) -> None:
        running = MagicMock(status="running")
        client = self._client(running)
        assert _adapter(client)._start() is True
        running.remove.assert_not_called()
        client.containers.run.assert_not_called()
