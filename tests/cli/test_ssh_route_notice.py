"""The CLI states which address it is about to dial, and its scope."""

from typing import Any

import pytest
from flowmesh_cli.commands import ssh as ssh_cmd


@pytest.fixture(autouse=True)
def _never_exec(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture the argv `_exec_ssh` would have replaced the process with."""
    calls: list[list[str]] = []
    monkeypatch.setattr(ssh_cmd.os, "execvp", lambda _bin, args: calls.append(args))
    monkeypatch.setattr(ssh_cmd.shutil, "which", lambda _name: "/usr/bin/ssh")
    return calls


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


def test_a_direct_connection_states_the_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ssh_cmd._exec_ssh(_info(), "tsk-1", None, direct=True)
    assert (
        "Direct route: 127.0.0.1:49393 (loopback on worker wkr-2)"
        in capsys.readouterr().err
    )


def test_a_degraded_session_states_the_route_without_the_direct_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No relay could serve it, so `host` is the only way in."""
    info = _info(mode="direct")
    info.pop("directHost")
    info.pop("directPort")
    ssh_cmd._exec_ssh(info, "tsk-1", None)
    assert (
        "Direct route: 127.0.0.1:49393 (loopback on worker wkr-2)"
        in capsys.readouterr().err
    )


def test_the_relay_route_is_not_described_as_direct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --direct a proxy session goes over the relay, not this address."""
    ssh_cmd._exec_ssh(_info(), "tsk-1", None)
    assert "Direct route" not in capsys.readouterr().err


def test_the_forward_listener_is_not_described_as_direct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --direct a forward session dials the server, not the worker."""
    info = _info(mode="forward", host="server.example.com", port=32001)
    ssh_cmd._exec_ssh(info, "tsk-1", None)
    assert "Direct route" not in capsys.readouterr().err


def test_a_worker_without_a_scope_says_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = _info(mode="direct")
    for key in ("directScope", "workerId", "directHost", "directPort"):
        info.pop(key, None)
    ssh_cmd._exec_ssh(info, "tsk-1", None)
    assert "Direct route" not in capsys.readouterr().err


def test_the_connection_still_happens(_never_exec: list[list[str]]) -> None:
    """Stating the scope informs; it never blocks the connection."""
    ssh_cmd._exec_ssh(_info(), "tsk-1", None, direct=True)
    assert _never_exec and "flowmesh@127.0.0.1" in _never_exec[0]
