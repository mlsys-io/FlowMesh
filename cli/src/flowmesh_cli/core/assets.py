"""Helpers for working with packaged CLI assets."""

from collections.abc import Iterable
from importlib import resources
from pathlib import Path


class AssetNotFoundError(FileNotFoundError):
    """Raised when a requested asset cannot be found."""


def asset_path(package: str, *parts: str) -> Path:
    """Resolve an asset path from an importable package.

    This returns a real path even when assets are inside a wheel by using
    importlib.resources.as_file.
    """
    resource = resources.files(package)
    for part in parts:
        resource /= part
    try:
        with resources.as_file(resource) as path:
            return Path(path)
    except FileNotFoundError as exc:
        raise AssetNotFoundError(str(resource)) from exc


def list_assets(package: str, relative_dir: Iterable[str] | None = None) -> list[Path]:
    """List available assets under a package directory."""
    resource = resources.files(package)
    if relative_dir:
        for part in relative_dir:
            resource /= part
    return [Path(item.name) for item in resource.iterdir()]
