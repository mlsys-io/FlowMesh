"""Tests for non-interactive SSH spec parsing and validation."""

import textwrap

import pytest

from server.task.parser import parse_workflow
from shared.tasks import TaskType
from shared.tasks.specs.ssh import SSHSpecStrict, SSHSpecTemplate

# ------------------------------------------------------------------ #
# Spec-level validation
# ------------------------------------------------------------------ #


def test_interactive_inferred_true_when_authorized_keys_present() -> None:
    spec = SSHSpecStrict.model_validate(
        {"taskType": "ssh", "authorizedKeys": ["ssh-ed25519 AAAA..."]}
    )
    assert spec.interactive is True


def test_interactive_defaults_false_when_no_signals() -> None:
    spec = SSHSpecStrict.model_validate({"taskType": "ssh"})
    assert spec.interactive is False


def test_interactive_inferred_false_when_command_set() -> None:
    spec = SSHSpecStrict.model_validate(
        {"taskType": "ssh", "image": "python:3.12", "command": ["python", "-c", "1"]}
    )
    assert spec.interactive is False


def test_interactive_inferred_false_when_entrypoint_set() -> None:
    spec = SSHSpecStrict.model_validate(
        {"taskType": "ssh", "image": "myimg:latest", "entrypoint": ["/run.sh"]}
    )
    assert spec.interactive is False


def test_interactive_explicit_false_accepted() -> None:
    spec = SSHSpecStrict.model_validate(
        {"taskType": "ssh", "interactive": False, "image": "myimg:latest"}
    )
    assert spec.interactive is False


def test_interactive_explicit_requires_authorized_keys() -> None:
    with pytest.raises(ValueError, match="authorizedKeys"):
        SSHSpecStrict.model_validate({"taskType": "ssh", "interactive": True})


def test_interactive_true_with_command_rejected() -> None:
    with pytest.raises(ValueError, match="interactive=true.*command/entrypoint"):
        SSHSpecStrict.model_validate(
            {
                "taskType": "ssh",
                "interactive": True,
                "command": ["echo", "hi"],
            }
        )


def test_interactive_true_with_entrypoint_rejected() -> None:
    with pytest.raises(ValueError, match="interactive=true.*command/entrypoint"):
        SSHSpecStrict.model_validate(
            {
                "taskType": "ssh",
                "interactive": True,
                "entrypoint": ["/bin/bash"],
            }
        )


def test_command_and_entrypoint_both_accepted() -> None:
    spec = SSHSpecStrict.model_validate(
        {
            "taskType": "ssh",
            "image": "myimg",
            "entrypoint": ["/bin/bash", "-c"],
            "command": ["echo hello"],
        }
    )
    assert spec.interactive is False
    assert spec.entrypoint == ["/bin/bash", "-c"]
    assert spec.command == ["echo hello"]


def test_template_interactive_inference() -> None:
    spec = SSHSpecTemplate.model_validate(
        {"taskType": "ssh", "image": "myimg", "command": ["ls"]}
    )
    assert spec.interactive is False


def test_noninteractive_with_inputs_valid() -> None:
    spec = SSHSpecStrict.model_validate(
        {
            "taskType": "ssh",
            "interactive": False,
            "image": "python:3.12",
            "command": ["python", "process.py"],
            "dependsOn": ["preprocess"],
            "inputs": [{"stage": "preprocess"}],
            "sshOutput": {"mountPath": "/mnt/flowmesh/output"},
        }
    )
    assert spec.interactive is False
    assert len(spec.inputs or []) == 1


def test_noninteractive_with_authorized_keys_lenient() -> None:
    """authorizedKeys are ignored for non-interactive, not rejected."""
    spec = SSHSpecStrict.model_validate(
        {
            "taskType": "ssh",
            "interactive": False,
            "image": "myimg",
            "command": ["echo"],
            "authorizedKeys": ["ssh-rsa AAAA..."],
        }
    )
    assert spec.interactive is False
    assert spec.authorizedKeys == ["ssh-rsa AAAA..."]


# ------------------------------------------------------------------ #
# Parser-level validation
# ------------------------------------------------------------------ #


def test_parse_noninteractive_ssh_workflow() -> None:
    """Non-interactive SSH tasks parse without requiring access mode validation."""
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: batch-wf
        spec:
          stages:
            - name: batch
              spec:
                taskType: ssh
                interactive: false
                image: python:3.12
                command: ["python", "-c", "print(1)"]
                ttlSeconds: 600
        """).strip()
    parsed = parse_workflow(payload, format="native")
    assert len(parsed.tasks) == 1
    task = parsed.tasks[0]
    assert task.task.spec.taskType == TaskType.SSH


def test_parse_noninteractive_ssh_skips_access_mode_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-interactive tasks don't fail when proxy/forward is disabled."""
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_SSH_PROXY", False)
    monkeypatch.setattr("server.task.parser._ENABLE_SERVER_PORT_FORWARD", False)

    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: batch-wf
        spec:
          stages:
            - name: run
              spec:
                taskType: ssh
                interactive: false
                image: myimg:latest
                command: ["./run.sh"]
        """).strip()
    # Should not raise even though proxy/forward are disabled.
    parsed = parse_workflow(payload, format="native")
    assert len(parsed.tasks) == 1
