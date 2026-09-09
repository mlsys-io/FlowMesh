"""Tests for the `external` worker provider and its shared-secret admission.

The property under test is the one the feature exists for: a token derived from
CONFIGURATION verifies after the supervisor has forgotten everything, whereas a
runtime-minted `uuid4()` token cannot.
"""

import pytest

from server.supervisor.adapters.external import (
    ExternalWorkerAdapter,
    ExternalWorkerConfig,
    ExternalWorkerFactory,
    mint_external_token,
    verify_external_token,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus

SECRET = "s3cret-shared-across-the-fleet"


class TestTokenVerification:
    def test_minted_token_verifies_and_yields_the_name(self) -> None:
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, SECRET) == "fm-worker-0"

    def test_token_is_deterministic(self) -> None:
        """The whole point: same inputs, same token, forever."""
        assert mint_external_token(SECRET, "w") == mint_external_token(SECRET, "w")

    def test_wrong_secret_is_rejected(self) -> None:
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, "not-the-secret") is None

    def test_tampered_name_is_rejected(self) -> None:
        """A caller cannot rename themselves into another worker's identity."""
        token = mint_external_token(SECRET, "fm-worker-0")
        _, _, digest = token.rpartition(".")
        assert verify_external_token(f"fm-worker-99.{digest}", SECRET) is None

    def test_names_containing_dots_round_trip(self) -> None:
        """The split is on the LAST dot, so a dotted name is not truncated."""
        name = "site.home.worker-3"
        assert verify_external_token(mint_external_token(SECRET, name), SECRET) == name

    @pytest.mark.parametrize("bad", ["", ".", "nodot", "name.", ".digest"])
    def test_malformed_tokens_return_none_rather_than_raise(self, bad: str) -> None:
        """Callers treat 'not external' and 'forged' identically, so every
        rejection path must return None instead of raising."""
        assert verify_external_token(bad, SECRET) is None

    def test_feature_is_off_when_no_secret_is_configured(self) -> None:
        """With no secret, a well-formed token from ANY secret admits nobody.

        This is what makes the change inert for existing deployments.
        """
        token = mint_external_token(SECRET, "fm-worker-0")
        assert verify_external_token(token, "") is None

    def test_token_survives_a_supervisor_restart(self) -> None:
        """The regression this feature exists to prevent.

        A runtime-minted token is only ever known to one process: a fresh
        registry (a restarted supervisor) cannot recognise it, which is how a
        healthy worker becomes permanently UNAUTHENTICATED. A config-derived
        token verifies against the secret alone, so it still proves identity
        after the process that first saw it is gone.
        """
        before = WorkerRegistry()
        runtime_token = before.new_token()
        token = mint_external_token(SECRET, "fm-worker-0")

        after_restart = WorkerRegistry()  # everything the old process knew is gone

        assert after_restart.try_get(runtime_token) is None
        assert verify_external_token(token, SECRET) == "fm-worker-0"


class TestExternalAdapter:
    def _adapter(self, name: str = "fm-worker-0") -> ExternalWorkerAdapter:
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        return factory.create_worker(
            mint_external_token(SECRET, name), ExternalWorkerConfig(), name=name
        )

    def test_starts_in_running_because_it_is_already_running(self) -> None:
        assert self._adapter().status is WorkerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_and_stop_are_successful_no_ops(self) -> None:
        """The supervisor neither launches nor kills an external worker, and
        must not report failure for work it was never supposed to do."""
        adapter = self._adapter()
        assert await adapter.start() is True
        assert await adapter.stop() is True

    def test_get_info_reports_the_external_provider_and_no_invented_hardware(
        self,
    ) -> None:
        info = self._adapter().get_info()
        assert info.provider == "external"
        assert info.name == "fm-worker-0"
        #: The supervisor cannot introspect a machine it does not own; a made-up
        #: profile would be fed straight to the scheduler.
        assert info.hardware is None

    def test_create_worker_refuses_a_token_with_no_verifiable_name(self) -> None:
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verifiable name"):
            factory.create_worker("garbage", ExternalWorkerConfig())  # type: ignore[arg-type]

    def test_destroy_does_not_pretend_to_stop_the_process(self) -> None:
        """Forgetting an external worker is accounting, not termination."""
        factory = ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]
        adapter = self._adapter()
        assert factory.destroy_worker(adapter) is None
        assert adapter.status is WorkerStatus.RUNNING


class TestDockerlessHost:
    """The supervisor must survive a host with no Docker daemon.

    `DockerWorkerFactory.__init__` acquires a client, so before this guard a
    dockerless host raised inside `WorkerManager.__init__` and killed the
    supervisor child -- while the FastAPI parent stayed up answering /healthz,
    which is what made it read Running/Ready with a dead worker plane.
    """

    def test_manager_builds_without_docker_and_keeps_external(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import logging

        from server.supervisor import manager as manager_mod
        from server.supervisor.registry import WorkerRegistry

        def _explode(_principal):
            raise RuntimeError("Error while fetching server API version")

        monkeypatch.setattr(manager_mod, "docker_provider_spec", _explode)
        monkeypatch.setattr(manager_mod, "vastai_provider_spec", _explode)

        mgr = manager_mod.WorkerManager(
            system_principal=None,  # type: ignore[arg-type]
            config_path=str(tmp_path / "absent.yaml"),
            registry=WorkerRegistry(),
            logger=logging.getLogger("test"),
        )

        #: The supervisor is alive and the external provider is usable, which is
        #: exactly the provider a dockerless host needs.
        assert "external" in mgr._providers
        assert "docker" not in mgr._providers
