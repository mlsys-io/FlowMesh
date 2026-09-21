"""Tests for the supervisor Docker adapter's handling of a concurrent remove."""

from types import SimpleNamespace
from typing import cast

from docker import DockerClient
from docker.errors import APIError, NotFound
from requests import Response

from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    _is_removal_in_progress,
)


def _api_error(status_code: int, explanation: str | None) -> APIError:
    # APIError.status_code is a read-only property over .response, so the status
    # has to arrive on a response object -- exactly as it does from the daemon.
    response = Response()
    response.status_code = status_code
    return APIError("boom", response=response, explanation=explanation)


class TestIsRemovalInProgress:
    def test_matches_the_concurrent_removal_409(self) -> None:
        exc = _api_error(409, "removal of container abc123 is already in progress")
        assert _is_removal_in_progress(exc) is True

    def test_rejects_other_409s(self) -> None:
        # 409 is also how Docker reports conflicts that do NOT resolve on their
        # own. Those must keep failing the start.
        name_taken = _api_error(
            409, 'Conflict. The container name "/gpu_0" is already in use'
        )
        running = _api_error(
            409, "You cannot remove a running container. Stop the container"
        )
        assert _is_removal_in_progress(name_taken) is False
        assert _is_removal_in_progress(running) is False

    def test_rejects_non_409_even_with_matching_text(self) -> None:
        assert _is_removal_in_progress(_api_error(500, "already in progress")) is False

    def test_tolerates_missing_explanation(self) -> None:
        # APIError.explanation is None when the daemon sent no body; must not raise.
        assert _is_removal_in_progress(_api_error(409, None)) is False


class _FakeContainers:
    """containers.get() raising NotFound after `gone_after` calls."""

    def __init__(self, gone_after: int) -> None:
        self.calls = 0
        self._gone_after = gone_after

    def get(self, _name: str) -> object:
        self.calls += 1
        if self.calls > self._gone_after:
            raise NotFound("gone")
        return SimpleNamespace()


def _adapter(containers: _FakeContainers) -> DockerWorkerAdapter:
    adapter = DockerWorkerAdapter.__new__(DockerWorkerAdapter)
    adapter._docker = cast(DockerClient, SimpleNamespace(containers=containers))
    adapter.container_name = "gpu_0"
    return adapter


class TestWaitContainerGone:
    def test_returns_true_once_the_container_disappears(self) -> None:
        containers = _FakeContainers(gone_after=2)
        adapter = _adapter(containers)
        assert adapter._wait_container_gone(timeout=5, poll=0) is True
        assert containers.calls == 3

    def test_returns_true_immediately_when_already_gone(self) -> None:
        containers = _FakeContainers(gone_after=0)
        assert _adapter(containers)._wait_container_gone(timeout=5, poll=0) is True

    def test_times_out_when_it_never_goes(self) -> None:
        # Bounded: the caller must be able to fail rather than hang forever.
        containers = _FakeContainers(gone_after=10**9)
        assert _adapter(containers)._wait_container_gone(timeout=0, poll=0) is False

    def test_keeps_waiting_through_a_transient_inspect_error(self) -> None:
        # Anything other than NotFound means "cannot confirm", so the deadline --
        # not an API blip -- is what ends the loop.
        class Flaky(_FakeContainers):
            def get(self, name: str) -> object:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transport blip")
                if self.calls > 2:
                    raise NotFound("gone")
                return SimpleNamespace()

        containers = Flaky(gone_after=0)
        assert _adapter(containers)._wait_container_gone(timeout=5, poll=0) is True
        assert containers.calls == 3
