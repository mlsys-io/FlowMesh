"""Tests for the FLOWMESH_PLUGINS loader in server.main.

The loader is a top-level snippet in main.py — too coupled to module-init
ordering to import that file in isolation. Instead, this test re-implements
the same expression inline and exercises it on temporary plugins to confirm:

  - empty / missing env var loads no plugins
  - whitespace and trailing commas are tolerated
  - each named module's install() is invoked and its HookBindings drained
    into the runtime registries (sync and async-ctxmgr forms)
  - install() returning a non-HookBindings raises TypeError
  - import errors propagate (loud failure, not silent skip)
"""

import importlib
import inspect
import os
import sys
from collections.abc import Iterator
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from lumid_hooks import HookBindings

from server.hooks import (
    IDENTITY_PROVIDERS,
    PERMISSION_CHECKERS,
    SUBMISSION_GUARDS,
    SUPPLIER_RESOLVERS,
    USAGE_SINKS,
    register,
)


# Mirrors the dispatch in src/server/main.py — kept verbatim so drift between
# the test and the real loader is obvious.
async def _load_plugins(stack: AsyncExitStack) -> None:
    raw = os.getenv("FLOWMESH_PLUGINS", "")
    for entry in raw.split(","):
        plugin_name = entry.strip()
        if not plugin_name:
            continue
        mod = importlib.import_module(plugin_name)
        rv = mod.install()
        if hasattr(rv, "__aenter__"):
            bindings = await stack.enter_async_context(rv)
        elif inspect.iscoroutine(rv):
            bindings = await rv
        else:
            bindings = rv
        if not isinstance(bindings, HookBindings):
            raise TypeError(
                f"{plugin_name}.install() must return HookBindings, got "
                f"{type(bindings).__name__}"
            )
        register(bindings)


@pytest.fixture
def plugin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Create a tmp dir on sys.path holding minimal plugin packages."""
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in (
        "alpha_plugin",
        "beta_plugin",
        "ctxmgr_plugin",
        "bad_return_plugin",
        "missing_install_plugin",
    ):
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    IDENTITY_PROVIDERS.clear()
    SUBMISSION_GUARDS.clear()
    USAGE_SINKS.clear()
    PERMISSION_CHECKERS.clear()
    SUPPLIER_RESOLVERS.clear()
    yield
    IDENTITY_PROVIDERS.clear()
    SUBMISSION_GUARDS.clear()
    USAGE_SINKS.clear()
    PERMISSION_CHECKERS.clear()
    SUPPLIER_RESOLVERS.clear()


def _write_plugin(root: Path, name: str, body: str) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body)


_SYNC_PLUGIN_BODY = """\
from flowmesh_hook import BaseBindings


class _Provider:
    name = "{name}"

    async def resolve(self, raw_token, logger):
        return None


def install():
    return BaseBindings(identity_providers=[_Provider()])
"""


class TestPluginLoader:
    @pytest.mark.anyio
    async def test_empty_env_loads_nothing(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        monkeypatch.delenv("FLOWMESH_PLUGINS", raising=False)
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
        assert IDENTITY_PROVIDERS == []

    @pytest.mark.anyio
    async def test_blank_env_loads_nothing(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        monkeypatch.setenv("FLOWMESH_PLUGINS", "")
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
        assert IDENTITY_PROVIDERS == []

    @pytest.mark.anyio
    async def test_sync_plugin_drains_into_registries(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        _write_plugin(
            plugin_dir, "alpha_plugin", _SYNC_PLUGIN_BODY.format(name="alpha")
        )
        monkeypatch.setenv("FLOWMESH_PLUGINS", "alpha_plugin")
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
        assert len(IDENTITY_PROVIDERS) == 1
        assert IDENTITY_PROVIDERS[0].name == "alpha"

    @pytest.mark.anyio
    async def test_multiple_plugins_drained_in_listed_order(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        _write_plugin(
            plugin_dir, "alpha_plugin", _SYNC_PLUGIN_BODY.format(name="alpha")
        )
        _write_plugin(plugin_dir, "beta_plugin", _SYNC_PLUGIN_BODY.format(name="beta"))
        monkeypatch.setenv("FLOWMESH_PLUGINS", "alpha_plugin , beta_plugin")
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
        assert [p.name for p in IDENTITY_PROVIDERS] == ["alpha", "beta"]

    @pytest.mark.anyio
    async def test_trailing_comma_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        _write_plugin(
            plugin_dir, "alpha_plugin", _SYNC_PLUGIN_BODY.format(name="alpha")
        )
        monkeypatch.setenv("FLOWMESH_PLUGINS", "alpha_plugin,,")
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
        assert [p.name for p in IDENTITY_PROVIDERS] == ["alpha"]

    @pytest.mark.anyio
    async def test_async_ctxmgr_plugin_drains_and_teardown_runs(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        body = """\
from contextlib import asynccontextmanager
from flowmesh_hook import BaseBindings


teardown_called = False


class _Provider:
    name = "ctx"
    async def resolve(self, raw_token, logger):
        return None


@asynccontextmanager
async def install():
    try:
        yield BaseBindings(identity_providers=[_Provider()])
    finally:
        global teardown_called
        teardown_called = True
"""
        _write_plugin(plugin_dir, "ctxmgr_plugin", body)
        monkeypatch.setenv("FLOWMESH_PLUGINS", "ctxmgr_plugin")
        async with AsyncExitStack() as stack:
            await _load_plugins(stack)
            assert [p.name for p in IDENTITY_PROVIDERS] == ["ctx"]
            mod = importlib.import_module("ctxmgr_plugin")
            assert mod.teardown_called is False
        assert importlib.import_module("ctxmgr_plugin").teardown_called is True

    @pytest.mark.anyio
    async def test_install_returning_non_bindings_raises(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        _write_plugin(
            plugin_dir,
            "bad_return_plugin",
            "def install():\n    return 'not-a-HookBindings'\n",
        )
        monkeypatch.setenv("FLOWMESH_PLUGINS", "bad_return_plugin")
        with pytest.raises(TypeError, match="must return HookBindings"):
            async with AsyncExitStack() as stack:
                await _load_plugins(stack)

    @pytest.mark.anyio
    async def test_missing_module_raises(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        monkeypatch.setenv("FLOWMESH_PLUGINS", "no_such_plugin_xyzzy")
        with pytest.raises(ModuleNotFoundError):
            async with AsyncExitStack() as stack:
                await _load_plugins(stack)

    @pytest.mark.anyio
    async def test_module_without_install_raises(
        self, monkeypatch: pytest.MonkeyPatch, plugin_dir: Path
    ) -> None:
        _write_plugin(plugin_dir, "missing_install_plugin", "pass\n")
        monkeypatch.setenv("FLOWMESH_PLUGINS", "missing_install_plugin")
        with pytest.raises(AttributeError):
            async with AsyncExitStack() as stack:
                await _load_plugins(stack)
