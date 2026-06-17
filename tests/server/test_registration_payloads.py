"""Tests for node registration payload construction."""

import pytest
from pydantic import ValidationError

from shared.schemas.node import NodeInfo


class TestNodeInfo:
    def test_required_fields(self) -> None:
        info = NodeInfo(
            namespace="prod",
            cluster="us-west",
            alias="node-01",
            version="0.1.0",
            started_at="2025-01-15T09:00:00Z",
            tags=["gpu", "a100"],
            last_seen="2025-01-15T10:30:00Z",
            max_gpu_count=8,
        )
        assert info.namespace == "prod"
        assert info.cluster == "us-west"
        assert info.alias == "node-01"
        assert info.version == "0.1.0"
        assert info.tags == ["gpu", "a100"]
        assert info.max_gpu_count == 8

    def test_roundtrip_json(self) -> None:
        info = NodeInfo(
            namespace="ns",
            cluster="cl",
            alias="grd",
            version="0.1.0",
            started_at="2025-01-01T00:00:00Z",
            tags=["gpu"],
            last_seen="2025-01-01T00:00:00Z",
            max_gpu_count=4,
        )
        restored = NodeInfo.model_validate_json(info.model_dump_json())
        assert restored == info

    def test_empty_tags(self) -> None:
        info = NodeInfo(
            namespace="ns",
            cluster="cl",
            alias="grd",
            version="0.1.0",
            started_at="2025-01-01T00:00:00Z",
            tags=[],
            last_seen="2025-01-01T00:00:00Z",
            max_gpu_count=0,
        )
        assert info.tags == []

    def test_version_optional(self) -> None:
        info = NodeInfo(
            namespace="ns",
            cluster="cl",
            alias="grd",
            started_at="2025-01-01T00:00:00Z",
            tags=[],
            last_seen="2025-01-01T00:00:00Z",
            max_gpu_count=0,
        )
        assert info.version is None

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            NodeInfo.model_validate(
                {
                    "namespace": "ns",
                    "cluster": "cl",
                    "started_at": "2025-01-01T00:00:00Z",
                    "tags": ["gpu"],
                    "last_seen": "2025-01-01T00:00:00Z",
                }
            )
