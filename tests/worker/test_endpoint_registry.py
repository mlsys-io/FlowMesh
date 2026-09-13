"""What a worker will relay to, and what it refuses."""

import logging
import threading

import pytest

from worker.relay import EndpointRegistry


class TestResolution:
    def test_published_endpoint_resolves_to_its_port(self) -> None:
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        assert registry.resolve("ssn-abc") == 2222

    def test_unknown_endpoint_resolves_to_none(self) -> None:
        """A relay for an id nobody published must not reach a port."""
        assert EndpointRegistry().resolve("ssn-never-published") is None

    def test_withdrawn_endpoint_stops_resolving(self) -> None:
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        registry.withdraw("ssn-abc")
        assert registry.resolve("ssn-abc") is None

    def test_endpoints_are_independent(self) -> None:
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        registry.publish("tsk-def", 8000)
        registry.withdraw("ssn-abc")
        assert registry.resolve("tsk-def") == 8000


class TestLifecycleEdges:
    def test_withdraw_is_idempotent(self) -> None:
        """Teardown runs on every exit path and must never raise."""
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        registry.withdraw("ssn-abc")
        registry.withdraw("ssn-abc")
        registry.withdraw("ssn-never-published")

    def test_republishing_replaces_the_port_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        with caplog.at_level(logging.WARNING, logger="worker.relay.registry"):
            registry.publish("ssn-abc", 3333)
        assert registry.resolve("ssn-abc") == 3333
        assert "ssn-abc" in caplog.text

    def test_republishing_the_same_port_is_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = EndpointRegistry()
        registry.publish("ssn-abc", 2222)
        with caplog.at_level(logging.WARNING, logger="worker.relay.registry"):
            registry.publish("ssn-abc", 2222)
        assert caplog.text == ""


class TestConcurrency:
    def test_publishers_and_resolvers_do_not_corrupt_the_table(self) -> None:
        """Executors publish from the main thread; relays resolve from their own."""
        registry = EndpointRegistry()
        errors: list[BaseException] = []
        start = threading.Event()

        def churn(index: int) -> None:
            start.wait()
            try:
                for _ in range(200):
                    registry.publish(f"ssn-{index}", 2000 + index)
                    assert registry.resolve(f"ssn-{index}") == 2000 + index
                    registry.withdraw(f"ssn-{index}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors
        assert all(registry.resolve(f"ssn-{i}") is None for i in range(8))
