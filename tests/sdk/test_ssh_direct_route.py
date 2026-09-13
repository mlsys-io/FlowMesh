"""Rendering a session's direct route and the scope it carries."""

from typing import Any

from flowmesh.ssh import (
    describe_direct_route,
    direct_route_scope,
    ssh_connection_commands,
)


def _info(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mode": "proxy",
        "username": "flowmesh",
        "host": "127.0.0.1",
        "port": 49393,
        "directHost": "127.0.0.1",
        "directPort": 49393,
        "directScope": "loopback",
        "workerId": "wkr-2",
    }
    base.update(over)
    return base


class TestDescribeDirectRoute:
    def test_a_loopback_route_names_the_worker_it_belongs_to(self) -> None:
        line = describe_direct_route(_info(), "127.0.0.1", 49393)
        assert line == "Direct route: 127.0.0.1:49393 (loopback on worker wkr-2)"

    def test_a_network_route_states_its_scope(self) -> None:
        """The docker backend's payload: network-bound, and still worker-owned."""
        info = _info(directScope="network")
        line = describe_direct_route(info, "worker-3.cluster.local", 49393)
        assert line == "Direct route: worker-3.cluster.local:49393 (network)"

    def test_the_worker_is_named_only_where_the_address_is_ambiguous(self) -> None:
        """A network address identifies its host; a loopback one cannot."""
        network = describe_direct_route(_info(directScope="network"), "w3", 49393)
        loopback = describe_direct_route(_info(), "127.0.0.1", 49393)
        assert network is not None and "wkr-2" not in network
        assert loopback is not None and "wkr-2" in loopback

    def test_a_loopback_route_without_a_worker_id_still_states_the_scope(self) -> None:
        line = describe_direct_route(_info(workerId=None), "127.0.0.1", 49393)
        assert line == "Direct route: 127.0.0.1:49393 (loopback)"

    def test_a_worker_that_publishes_no_scope_says_nothing(self) -> None:
        assert describe_direct_route({}, "127.0.0.1", 49393) is None

    def test_an_incomplete_address_says_nothing(self) -> None:
        assert describe_direct_route(_info(), "127.0.0.1", None) is None
        assert describe_direct_route(_info(), None, 49393) is None


class TestRenderedCommandLabels:
    def test_a_relayed_session_labels_its_direct_command_with_the_scope(self) -> None:
        labels = dict(ssh_connection_commands("tsk-1", _info(), "http://s:8000"))
        assert "ssh (direct, loopback on worker wkr-2)" in labels

    def test_the_label_and_the_cli_line_use_the_same_phrase(self) -> None:
        """Two surfaces, one wording -- they cannot drift apart."""
        info = _info()
        line = describe_direct_route(info, "127.0.0.1", 49393)
        labels = dict(ssh_connection_commands("tsk-1", info, "http://s:8000"))
        phrase = direct_route_scope(info)
        assert line is not None and f"({phrase})" in line
        assert any(
            f"({phrase})" in label or f", {phrase})" in label for label in labels
        )

    def test_a_degraded_session_labels_its_only_command_with_the_scope(self) -> None:
        """`direct` here is the sole route, so the scope matters most."""
        info = _info(mode="direct")
        info.pop("directHost")
        info.pop("directPort")
        labels = dict(ssh_connection_commands("tsk-1", info, "http://s:8000"))
        assert "ssh (loopback on worker wkr-2)" in labels

    def test_a_network_session_labels_its_direct_command_too(self) -> None:
        info = _info(directScope="network", host="w3", directHost="w3")
        labels = dict(ssh_connection_commands("tsk-1", info, "http://s:8000"))
        assert "ssh (direct, network)" in labels

    def test_a_worker_without_a_scope_keeps_the_plain_labels(self) -> None:
        info = _info(mode="direct")
        for key in ("directHost", "directPort", "directScope", "workerId"):
            info.pop(key, None)
        labels = dict(ssh_connection_commands("tsk-1", info, "http://s:8000"))
        assert "ssh" in labels
