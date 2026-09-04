"""Version helpers for the FlowMesh CLI."""

from importlib.metadata import PackageNotFoundError, version

_PACKAGE_NAME = "flowmesh-cli"
_STATIC_VERSION = "0.1.8"


def _resolve_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _STATIC_VERSION


__version__ = _resolve_version()
