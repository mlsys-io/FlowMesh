"""Version helpers for the FlowMesh CLI."""

from importlib.metadata import PackageNotFoundError, version


def resolve_cli_version() -> str:
    """Return the installed flowmesh-cli version, or ``"unknown"`` if unreadable."""
    try:
        return version("flowmesh-cli")
    except PackageNotFoundError:
        return "unknown"
