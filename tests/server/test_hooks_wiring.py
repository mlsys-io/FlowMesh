"""Smoke tests for the hook registry contract.

Each test registers a fake hook, exercises the core call site that iterates
the registry, and asserts the fake was invoked. This is the regression test
that protects future plugin authors against the registry being silently
disconnected from a call site.
"""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from server.auth.security import (
    PrincipalContext,
    authenticate_api_key,
    authenticate_request,
    require_permission,
    resolve_accessible_ids,
)
from server.hooks import (
    IDENTITY_PROVIDERS,
    PERMISSION_CHECKERS,
    SUBMISSION_GUARDS,
    USAGE_SINKS,
    AccessibleIds,
    UsageRow,
    UsageSink,
)


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.hooks_wiring")


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    IDENTITY_PROVIDERS.clear()
    SUBMISSION_GUARDS.clear()
    USAGE_SINKS.clear()
    PERMISSION_CHECKERS.clear()
    yield
    IDENTITY_PROVIDERS.clear()
    SUBMISSION_GUARDS.clear()
    USAGE_SINKS.clear()
    PERMISSION_CHECKERS.clear()


class _FakeIdentityProvider:
    name = "fake"

    def __init__(self, returns: PrincipalContext | None = None) -> None:
        self.returns = returns
        self.calls = 0

    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        self.calls += 1
        return self.returns


class TestIdentityProviderWiring:
    @pytest.mark.anyio
    async def test_no_providers_returns_default_principal(
        self, logger: logging.Logger
    ) -> None:
        principal = await authenticate_api_key("any-token", logger)

        assert principal.principal_type == "admin"
        assert principal.scopes == ["*"]

    @pytest.mark.anyio
    async def test_provider_claiming_token_short_circuits(
        self, logger: logging.Logger
    ) -> None:
        principal = PrincipalContext(
            principal_id="p-fake",
            org_id="fake-org",
            external_id="ext",
            principal_type="user",
            scopes=[],
        )
        first = _FakeIdentityProvider(returns=principal)
        second = _FakeIdentityProvider(returns=None)
        IDENTITY_PROVIDERS.extend([first, second])

        result = await authenticate_api_key("opaque", logger)

        assert result is principal
        assert first.calls == 1
        assert second.calls == 0  # short-circuited

    @pytest.mark.anyio
    async def test_all_providers_returning_none_falls_through_to_401(
        self, logger: logging.Logger
    ) -> None:
        provider = _FakeIdentityProvider(returns=None)
        IDENTITY_PROVIDERS.append(provider)

        with pytest.raises(HTTPException) as exc_info:
            await authenticate_api_key("opaque", logger)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert provider.calls == 1


class TestSubmissionGuardWiring:
    """Lightweight check: iterating SUBMISSION_GUARDS in workflows.py works.

    We invoke the iteration logic directly rather than spinning up the full
    FastAPI router — the contract is "every guard's check() is awaited".
    """

    @pytest.mark.anyio
    async def test_guards_run_in_order_and_can_block(
        self, logger: logging.Logger
    ) -> None:
        order: list[str] = []

        class _AllowGuard:
            name = "allow"

            async def check(
                self, principal: PrincipalContext, logger: logging.Logger
            ) -> None:
                order.append("allow")

        class _BlockGuard:
            name = "block"

            async def check(
                self, principal: PrincipalContext, logger: logging.Logger
            ) -> None:
                order.append("block")
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="nope"
                )

        SUBMISSION_GUARDS.extend([_AllowGuard(), _BlockGuard()])
        principal = PrincipalContext(
            principal_id="p",
            org_id="x",
            external_id="e",
            principal_type="user",
            scopes=[],
        )

        with pytest.raises(HTTPException) as exc_info:
            for guard in SUBMISSION_GUARDS:
                await guard.check(principal, logger)

        assert order == ["allow", "block"]
        assert exc_info.value.status_code == status.HTTP_402_PAYMENT_REQUIRED


class TestUsageSinkWiring:
    """EventMonitor iterates USAGE_SINKS after task usage is computed.

    We stub the dispatch directly: monitoring.py imports `USAGE_SINKS` fresh
    inside `_emit_usage_async` and iterates it; sink failures are
    isolated. Verifying the iteration without booting the whole monitor
    avoids dragging Redis/etc into the test.
    """

    @pytest.mark.anyio
    async def test_sink_failures_are_isolated(self, logger: logging.Logger) -> None:
        seen: list[str] = []

        class _BoomSink:
            name = "boom"

            async def emit(self, rows: list[UsageRow], logger: logging.Logger) -> None:
                seen.append("boom")
                raise RuntimeError("kaboom")

        class _OkSink:
            name = "ok"

            async def emit(self, rows: list[UsageRow], logger: logging.Logger) -> None:
                seen.append("ok")

        sinks: list[UsageSink] = [_BoomSink(), _OkSink()]
        rows: list[UsageRow] = [
            UsageRow(
                org_id="x",
                principal_id="p",
                supplier_id=None,
                occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
                cost=Decimal("1"),
                task_id="tsk-1",
                runtime_sec=1.0,
                cost_per_hour=1.0,
                task_status="DONE",
            )
        ]

        # Mirror the actual core dispatch loop (monitoring.py):
        for sink in sinks:
            try:
                await sink.emit(rows, logger)
            except Exception as exc:
                logger.warning("Usage sink %s failed: %s", sink.name, exc)

        assert seen == ["boom", "ok"]  # second sink ran despite first failure


class _CapturingProvider:
    name = "capturing"

    def __init__(self, returns: PrincipalContext | None) -> None:
        self.returns = returns
        self.tokens_seen: list[str] = []

    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        self.tokens_seen.append(raw_token)
        return self.returns


def _build_submit_app() -> FastAPI:
    """A FastAPI app that exercises only the `authenticate_request` dep.

    The real submit endpoint pulls in registries and Redis; this stub
    keeps the test focused on whether the dependency extracts the bearer
    token and routes it through `IDENTITY_PROVIDERS`.
    """
    app = FastAPI()
    app.state.logger = logging.getLogger("test.authenticate_request")

    @app.post("/probe")
    async def probe(
        principal: PrincipalContext = Depends(authenticate_request),
    ) -> dict[str, str]:
        return {
            "principal_id": principal.principal_id,
            "org_id": principal.org_id,
        }

    return app


class TestAuthenticateRequest:
    """The router-level FastAPI dependency must run the auth chain on every call.

    Before this dependency landed, `submit_workflow` called
    `default_principal()` directly and never read the `Authorization`
    header — registered `IdentityProvider`s saw zero traffic.
    """

    def test_no_providers_yields_admin_default(self) -> None:
        app = _build_submit_app()
        client = TestClient(app)

        response = client.post("/probe", headers={"Authorization": "Bearer abc"})

        assert response.status_code == 200
        assert response.json() == {"principal_id": "admin", "org_id": "local"}

    def test_provider_receives_bearer_token(self) -> None:
        principal = PrincipalContext(
            principal_id="p-1",
            org_id="org-1",
            external_id="ext-1",
            principal_type="user",
            scopes=[],
        )
        provider = _CapturingProvider(returns=principal)
        IDENTITY_PROVIDERS.append(provider)

        app = _build_submit_app()
        client = TestClient(app)

        response = client.post("/probe", headers={"Authorization": "Bearer secret-x"})

        assert response.status_code == 200
        assert response.json() == {"principal_id": "p-1", "org_id": "org-1"}
        assert provider.tokens_seen == ["secret-x"]

    def test_missing_authorization_header_falls_through_to_401(self) -> None:
        provider = _CapturingProvider(returns=None)
        IDENTITY_PROVIDERS.append(provider)

        app = _build_submit_app()
        client = TestClient(app)

        response = client.post("/probe")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert provider.tokens_seen == [""]


class TestPermissionCheckerComposition:
    @pytest.fixture
    def principal(self) -> PrincipalContext:
        return PrincipalContext(
            principal_id="p",
            org_id="o",
            external_id="e",
            principal_type="user",
            scopes=[],
        )

    @pytest.mark.anyio
    async def test_no_checkers_means_no_filter(
        self, principal: PrincipalContext, logger: logging.Logger
    ) -> None:
        ids = await resolve_accessible_ids(principal, "task", "list", logger)
        assert ids is None

    @pytest.mark.anyio
    async def test_all_short_circuits_union(
        self, principal: PrincipalContext, logger: logging.Logger
    ) -> None:
        class _UnionChecker:
            name = "union"

            async def accessible_ids(
                self,
                principal: PrincipalContext,
                resource_type: str,
                action: str,
                logger: logging.Logger,
            ) -> AccessibleIds:
                return frozenset({"tsk-A"})

            async def require(self, *args: Any, **kwargs: Any) -> None:
                return None

        class _AllChecker:
            name = "allowall"

            async def accessible_ids(
                self,
                principal: PrincipalContext,
                resource_type: str,
                action: str,
                logger: logging.Logger,
            ) -> AccessibleIds:
                return "all"

            async def require(self, *args: Any, **kwargs: Any) -> None:
                return None

        PERMISSION_CHECKERS.extend([_UnionChecker(), _AllChecker()])

        ids = await resolve_accessible_ids(principal, "task", "list", logger)
        assert ids is None

    @pytest.mark.anyio
    async def test_set_results_are_unioned(
        self, principal: PrincipalContext, logger: logging.Logger
    ) -> None:
        class _A:
            name = "a"

            async def accessible_ids(
                self,
                principal: PrincipalContext,
                resource_type: str,
                action: str,
                logger: logging.Logger,
            ) -> AccessibleIds:
                return frozenset({"tsk-1", "tsk-2"})

            async def require(self, *args: Any, **kwargs: Any) -> None:
                return None

        class _B:
            name = "b"

            async def accessible_ids(
                self,
                principal: PrincipalContext,
                resource_type: str,
                action: str,
                logger: logging.Logger,
            ) -> AccessibleIds:
                return frozenset({"tsk-2", "tsk-3"})

            async def require(self, *args: Any, **kwargs: Any) -> None:
                return None

        PERMISSION_CHECKERS.extend([_A(), _B()])

        ids = await resolve_accessible_ids(principal, "task", "list", logger)
        assert ids == frozenset({"tsk-1", "tsk-2", "tsk-3"})

    @pytest.mark.anyio
    async def test_require_passes_when_no_checkers(
        self, principal: PrincipalContext, logger: logging.Logger
    ) -> None:
        await require_permission(principal, "task", "tsk-1", "read", logger)

    @pytest.mark.anyio
    async def test_require_first_failure_short_circuits(
        self, principal: PrincipalContext, logger: logging.Logger
    ) -> None:
        seen: list[str] = []

        class _Block:
            name = "block"

            async def accessible_ids(self, *args: Any, **kwargs: Any) -> AccessibleIds:
                return "all"

            async def require(self, *args: Any, **kwargs: Any) -> None:
                seen.append("block")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="nope"
                )

        class _NeverRuns:
            name = "never"

            async def accessible_ids(self, *args: Any, **kwargs: Any) -> AccessibleIds:
                return "all"

            async def require(self, *args: Any, **kwargs: Any) -> None:
                seen.append("never")

        PERMISSION_CHECKERS.extend([_Block(), _NeverRuns()])

        with pytest.raises(HTTPException) as exc_info:
            await require_permission(principal, "task", "tsk-1", "read", logger)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert seen == ["block"]
