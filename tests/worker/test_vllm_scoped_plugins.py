"""Tests for VLLMExecutor's vllm-omni plugin exclusion via VLLM_PLUGINS."""

import importlib.metadata
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")

from worker.executors.vllm_executor import VLLMExecutor  # noqa: E402

_OMNI_EP_NAME = "vllm_omni_register_models"
_OMNI_EP_VALUE = "vllm_omni.engine.arg_utils:register_omni_models_to_vllm"


def _ep(name: str, value: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(
        name=name, value=value, group="vllm.general_plugins"
    )


def _mock_eps(
    monkeypatch: pytest.MonkeyPatch, eps: list[importlib.metadata.EntryPoint]
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: eps if group == "vllm.general_plugins" else [],
    )


class TestPluginAllowlist:
    def test_excludes_omni_keeps_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        discovered = [
            _ep(_OMNI_EP_NAME, _OMNI_EP_VALUE),
            _ep("some_other_plugin", "some_pkg.plugins:register"),
        ]
        _mock_eps(monkeypatch, discovered)
        result = VLLMExecutor._vllm_plugins_allowlist_excluding_omni()
        assert result == "some_other_plugin"

    def test_empty_when_only_omni(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_eps(monkeypatch, [_ep(_OMNI_EP_NAME, _OMNI_EP_VALUE)])
        # "" is a valid allowlist (load none); NOT None which would mean load-all.
        assert VLLMExecutor._vllm_plugins_allowlist_excluding_omni() == ""

    def test_filters_by_module_not_ep_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discovered = [
            _ep("renamed_entry", "vllm_omni.something.else:fn"),
            _ep("keep", "another_pkg:fn"),
        ]
        _mock_eps(monkeypatch, discovered)
        assert VLLMExecutor._vllm_plugins_allowlist_excluding_omni() == "keep"

    def test_real_install_does_not_allowlist_omni(self) -> None:
        eps = importlib.metadata.entry_points(group="vllm.general_plugins")
        if not any(
            ep.value.split(":", 1)[0].strip().startswith("vllm_omni") for ep in eps
        ):
            pytest.skip("vllm_omni not a registered vllm.general_plugins entry")
        names = VLLMExecutor._vllm_plugins_allowlist_excluding_omni().split(",")
        assert _OMNI_EP_NAME not in names


class TestAllowlistHelperIsPure:
    def test_helper_does_not_mutate_env(self) -> None:
        before = os.environ.get("VLLM_PLUGINS")
        VLLMExecutor._vllm_plugins_allowlist_excluding_omni()
        assert os.environ.get("VLLM_PLUGINS") == before


_OMNI_MODULE_PREFIX = "vllm_omni"


class TestPluginScopingPreventsRebind:
    def test_allowlist_keeps_vllm_omni_unloaded(self) -> None:
        """VLLM_PLUGINS fix allowlist keeps vllm_omni unloaded and Request unpatched."""

        def _is_omni_ep(ep: importlib.metadata.EntryPoint) -> bool:
            return ep.value.split(":", 1)[0].strip().startswith(_OMNI_MODULE_PREFIX)

        eps = importlib.metadata.entry_points(group="vllm.general_plugins")
        if not any(_is_omni_ep(ep) for ep in eps):
            pytest.skip("vllm_omni not a registered vllm.general_plugins entry")

        allowlist = VLLMExecutor._vllm_plugins_allowlist_excluding_omni()
        script = "\n".join(
            [
                "import sys, os",
                f"os.environ['VLLM_PLUGINS'] = {allowlist!r}",
                "try:",
                "    from vllm.plugins import load_general_plugins",
                "    load_general_plugins()",
                "except Exception:",
                "    pass",
                "import vllm.v1.request as R",
                "print('omni_loaded:', 'vllm_omni' in sys.modules)",
                "print('request_module:', R.Request.__module__)",
            ]
        )
        result = subprocess.run(  # nosec B603 — fixed argv list, no shell=True, sys.executable
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "omni_loaded: False" in result.stdout
        assert "request_module: vllm.v1.request" in result.stdout


def _subprocess_env() -> dict[str, str]:
    """Build an env with src/ on PYTHONPATH so subprocess can import worker.*."""
    src_dir = (pathlib.Path(__file__).parents[2] / "src").as_posix()
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = f"{src_dir}:{existing}" if existing else src_dir
    return {**os.environ, "PYTHONPATH": pythonpath}


class TestLazyOmniRegistry:
    """Regression tests: import worker.executors must not pull in vllm_omni."""

    _OMNI_KEYS = (
        "omni_text2image",
        "omni_text2speech",
        "omni_text2audio",
        "omni_text2general",
    )

    def test_executor_registry_has_omni_keys(self) -> None:
        from worker.executors import EXECUTOR_REGISTRY

        for key in self._OMNI_KEYS:
            assert key in EXECUTOR_REGISTRY

    def test_import_does_not_load_vllm_omni(self) -> None:
        """import worker.executors alone must not load vllm_omni or patch Request."""
        script = "\n".join(
            [
                "import sys",
                "import worker.executors",
                "import vllm.v1.request as R",
                "print('omni_in_modules:', 'vllm_omni' in sys.modules)",
                "print('request_class:', R.Request.__name__)",
            ]
        )
        result = subprocess.run(  # nosec B603 — fixed argv list, no shell=True, sys.executable
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr
        assert "omni_in_modules: False" in result.stdout
        assert "request_class: Request" in result.stdout
