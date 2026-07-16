"""Tests for serve-task accessMode validation at submission time."""

import textwrap

import pytest

from server.task.parser import parse_workflow


def _serve_workflow(access_mode: str) -> str:
    return textwrap.dedent(f"""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: serve-wf
        spec:
          stages:
            - name: serve
              spec:
                taskType: serve
                accessMode: {access_mode}
                resources:
                  hardware:
                    gpu:
                      type: any
                      count: 1
                model:
                  source:
                    type: huggingface
                    identifier: Qwen/Qwen3-0.6B
        """).strip()


def test_parse_serve_proxy_rejected_when_proxy_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_PROXY", False)
    with pytest.raises(ValueError, match="serve accessMode 'proxy' is disabled"):
        parse_workflow(_serve_workflow("proxy"), format="native")


def test_parse_serve_proxy_ignores_ssh_proxy_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SSH_PROXY", False)
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_PROXY", True)

    parsed = parse_workflow(_serve_workflow("proxy"), format="native")

    assert len(parsed.tasks) == 1


def test_parse_serve_forward_rejected_when_forward_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_PORT_FORWARD", False)
    with pytest.raises(ValueError, match="serve accessMode 'forward' is disabled"):
        parse_workflow(_serve_workflow("forward"), format="native")


def test_parse_serve_direct_ok_when_both_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SERVE_PROXY", False)
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_PORT_FORWARD", False)
    parsed = parse_workflow(_serve_workflow("direct"), format="native")
    assert len(parsed.tasks) == 1


def test_parse_serve_proxy_ok_when_enabled() -> None:
    parsed = parse_workflow(_serve_workflow("proxy"), format="native")
    assert len(parsed.tasks) == 1
