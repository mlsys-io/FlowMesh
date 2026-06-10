"""Tests for worker-alias construction in ``flowmesh_stack.workers``."""

import json
from typing import Any

import pytest
from flowmesh.exceptions import FlowMeshError
from flowmesh_stack import workers


def _aliases(payloads: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for payload, _label in payloads:
        obj: dict[str, Any] = json.loads(payload)
        out.append(obj["worker_config"]["worker_alias"])
    return out


def _cpu_payloads(
    count: int, slug: str | None = None, name_template: str | None = None
) -> list[tuple[str, str]]:
    return workers._payloads_for_worker_create(
        kind="cpu",
        count=count,
        targets="all",
        config_paths=None,
        config_raw=None,
        name_template=name_template,
        slug=slug,
    )


def test_default_cpu_aliases_are_slug_prefixed() -> None:
    assert _aliases(_cpu_payloads(2, slug="flowmesh_node")) == [
        "flowmesh_node_worker_cpu_0",
        "flowmesh_node_worker_cpu_1",
    ]


def test_no_slug_reproduces_legacy_bare_names() -> None:
    assert _aliases(_cpu_payloads(2)) == ["worker_cpu_0", "worker_cpu_1"]


def test_template_renders_placeholders() -> None:
    payloads = _cpu_payloads(2, slug="flowmesh_node", name_template="{slug}-run-{idx}")
    assert _aliases(payloads) == ["flowmesh_node-run-0", "flowmesh_node-run-1"]


def test_template_kind_placeholder() -> None:
    payloads = _cpu_payloads(1, slug="s", name_template="{kind}-{idx}")
    assert _aliases(payloads) == ["cpu-0"]


def test_unknown_placeholder_raises_with_available_fields() -> None:
    with pytest.raises(FlowMeshError) as exc:
        _cpu_payloads(1, name_template="{bogus}")
    message = str(exc.value)
    assert "bogus" in message
    assert "{slug}, {kind}, {idx}, {gpu}" in message


def test_malformed_template_raises_flowmesh_error() -> None:
    with pytest.raises(FlowMeshError, match="invalid --name-template"):
        _cpu_payloads(1, name_template="{slug")


def test_template_missing_index_collides() -> None:
    with pytest.raises(FlowMeshError, match="duplicate worker names"):
        _cpu_payloads(2, name_template="{slug}-static")


def test_gpu_aliases_default_and_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "detect_gpu_targets", lambda _targets: ["0", "1"])

    default_payloads = workers._payloads_for_worker_create(
        kind="gpu",
        count=2,
        targets="0,1",
        config_paths=None,
        config_raw=None,
        slug="flowmesh_node",
    )
    assert _aliases(default_payloads) == [
        "flowmesh_node_worker_gpu_0",
        "flowmesh_node_worker_gpu_1",
    ]

    templated = workers._payloads_for_worker_create(
        kind="gpu",
        count=2,
        targets="0,1",
        config_paths=None,
        config_raw=None,
        name_template="g{gpu}",
        slug="flowmesh_node",
    )
    assert _aliases(templated) == ["g0", "g1"]


def test_gpu_single_worker_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "detect_gpu_targets", lambda _targets: ["0", "1"])
    payloads = workers._payloads_for_worker_create(
        kind="gpu",
        count=1,
        targets="0,1",
        config_paths=None,
        config_raw=None,
        slug="flowmesh_node",
    )
    assert _aliases(payloads) == ["flowmesh_node_worker_gpu_0_1"]
