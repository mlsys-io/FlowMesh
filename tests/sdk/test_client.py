"""Client construction and configuration tests."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh import AsyncFlowMesh, ConfigInvalidError, FlowMesh, FlowMeshConfig
from flowmesh.client import resolve_config

_EXPECTED_RESOURCES = [
    "results",
    "nodes",
    "ssh",
    "system",
    "tasks",
    "workers",
    "workflows",
]


class TestFlowMeshClient:
    def test_explicit_params(self) -> None:
        client = FlowMesh(base_url="http://example.com:8000", api_key="flm-test")
        assert client.base_url == "http://example.com:8000"
        assert client.api_key == "flm-test"

    def test_all_resource_namespaces_present(self) -> None:
        client = FlowMesh(base_url="http://localhost:8000")
        for attr in _EXPECTED_RESOURCES:
            assert hasattr(client, attr), f"Missing resource namespace: {attr}"

    def test_env_var_resolution(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://env-host:9000",
                "FLOWMESH_API_KEY": "flm-env-key",
            },
        ):
            client = FlowMesh()
            assert client.base_url == "http://env-host:9000"
            assert client.api_key == "flm-env-key"

    def test_explicit_overrides_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://env-host:9000",
                "FLOWMESH_API_KEY": "flm-env-key",
            },
        ):
            client = FlowMesh(base_url="http://explicit:8000", api_key="flm-explicit")
            assert client.base_url == "http://explicit:8000"
            assert client.api_key == "flm-explicit"

    def test_default_base_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for key in ("FLOWMESH_BASE_URL", "FLOWMESH_API_KEY"):
                os.environ.pop(key, None)
            client = FlowMesh()
            assert "localhost" in client.base_url or "8000" in client.base_url

    def test_context_manager(self) -> None:
        with FlowMesh(base_url="http://localhost:8000") as client:
            assert client.base_url == "http://localhost:8000"

    def test_trailing_slash_stripped(self) -> None:
        client = FlowMesh(base_url="http://example.com:8000/")
        assert client.base_url == "http://example.com:8000"


class TestAsyncFlowMeshClient:
    def test_explicit_params(self) -> None:
        client = AsyncFlowMesh(base_url="http://example.com:8000", api_key="flm-test")
        assert client.base_url == "http://example.com:8000"

    def test_all_resource_namespaces_present(self) -> None:
        client = AsyncFlowMesh(base_url="http://localhost:8000")
        for attr in _EXPECTED_RESOURCES:
            assert hasattr(client, attr), f"Missing resource namespace: {attr}"

    def test_env_var_resolution(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://env-host:9000",
                "FLOWMESH_API_KEY": "flm-env-key",
            },
        ):
            client = AsyncFlowMesh()
            assert client.base_url == "http://env-host:9000"
            assert client.api_key == "flm-env-key"

    def test_explicit_overrides_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://env-host:9000",
                "FLOWMESH_API_KEY": "flm-env-key",
            },
        ):
            client = AsyncFlowMesh(
                base_url="http://explicit:8000", api_key="flm-explicit"
            )
            assert client.base_url == "http://explicit:8000"
            assert client.api_key == "flm-explicit"

    def test_default_base_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for key in ("FLOWMESH_BASE_URL", "FLOWMESH_API_KEY"):
                os.environ.pop(key, None)
            client = AsyncFlowMesh()
            assert "localhost" in client.base_url or "8000" in client.base_url

    @pytest.mark.anyio
    async def test_context_manager(self) -> None:
        async with AsyncFlowMesh(base_url="http://localhost:8000") as client:
            assert client.base_url == "http://localhost:8000"

    def test_trailing_slash_stripped(self) -> None:
        client = AsyncFlowMesh(base_url="http://example.com:8000/")
        assert client.base_url == "http://example.com:8000"


class TestFlowMeshConfig:
    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://cfg-host:8000",
                "FLOWMESH_API_KEY": "flm-cfg-key",
            },
        ):
            cfg = FlowMeshConfig.from_env()
            assert cfg.base_url == "http://cfg-host:8000"
            assert cfg.api_key == "flm-cfg-key"

    def test_from_env_missing_url_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FLOWMESH_BASE_URL", None)
            with pytest.raises(Exception):
                FlowMeshConfig.from_env()

    def test_from_env_no_api_key_returns_none(self) -> None:
        with patch.dict(
            os.environ, {"FLOWMESH_BASE_URL": "http://cfg-host:8000"}, clear=True
        ):
            cfg = FlowMeshConfig.from_env()
            assert cfg.api_key is None

    def test_from_mapping_empty_base_url_raises(self) -> None:
        with pytest.raises(ConfigInvalidError):
            FlowMeshConfig.from_mapping({"base_url": ""})


class TestResolveConfig:
    def test_both_env_vars_skips_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            'base_url = "http://file-host:8000"\napi_key = "file-key"\n'
        )

        with patch.dict(
            os.environ,
            {
                "FLOWMESH_BASE_URL": "http://env-host:8000",
                "FLOWMESH_API_KEY": "env-key",
            },
            clear=True,
        ):
            cfg = resolve_config(config_path=config_file)

        assert cfg.base_url == "http://env-host:8000"
        assert cfg.api_key == "env-key"

    def test_base_url_env_merges_api_key_from_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            'base_url = "http://file-host:8000"\napi_key = "file-key"\n'
        )

        with patch.dict(
            os.environ,
            {"FLOWMESH_BASE_URL": "http://env-host:8000"},
            clear=True,
        ):
            cfg = resolve_config(config_path=config_file)

        assert cfg.base_url == "http://env-host:8000"
        assert cfg.api_key == "file-key"
