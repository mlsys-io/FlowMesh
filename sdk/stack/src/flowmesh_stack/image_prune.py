"""Policy-based selection for ``flowmesh stack image prune``.

Pure logic over :class:`~flowmesh_stack.docker.ManagedImage` lists: the caller
supplies the discovered images (with ``in_use`` already populated) and the
policy flags; :func:`select_prune_targets` returns which images to delete and
which are protected. Docker is never touched here.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .docker import ManagedImage

_DURATION = re.compile(r"^(\d+)([smhdw])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> timedelta:
    """Parse a ``<int><unit>`` duration (units ``s``/``m``/``h``/``d``/``w``)."""
    match = _DURATION.match(text.strip())
    if not match:
        raise ValueError(
            f"invalid duration {text!r}; expected an integer followed by one of "
            "s, m, h, d, w (e.g. 30d, 12h)"
        )
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])


@dataclass
class PrunePlan:
    """Result of a prune selection: what to delete and what is protected."""

    deleted: list[ManagedImage] = field(default_factory=list)
    protected: list[tuple[ManagedImage, str]] = field(default_factory=list)


def _keep_last_versions(
    parsed: list[ManagedImage], keep_last: int
) -> dict[str, set[str]]:
    newest: dict[str, dict[str, datetime]] = {}
    for image in parsed:
        if image.target is None or image.version is None:
            continue
        per_target = newest.setdefault(image.target, {})
        if image.version not in per_target or image.created > per_target[image.version]:
            per_target[image.version] = image.created
    protected: dict[str, set[str]] = {}
    for target, versions in newest.items():
        ordered = sorted(versions.items(), key=lambda item: item[1], reverse=True)
        protected[target] = {version for version, _ in ordered[:keep_last]}
    return protected


def select_prune_targets(
    images: list[ManagedImage],
    *,
    keep_last: int | None = None,
    keep_versions: set[str] | None = None,
    keep_active: bool = False,
    older_than: timedelta | None = None,
    include_dangling: bool = False,
    now: datetime,
) -> PrunePlan:
    """Select images to prune as ``candidate pool − protected``.

    The candidate pool is a filter — protections only subtract from it. The
    non-dangling base is restricted to parsed images (``target`` set) so manual
    tags on a managed repo are never candidates.
    """
    keep_versions = keep_versions or set()
    parsed = [i for i in images if not i.dangling and i.target is not None]
    dangling = [i for i in images if i.dangling]

    if older_than is not None:
        cutoff = now - older_than
        pool: list[ManagedImage] = [i for i in parsed if i.created < cutoff]
    elif keep_last is not None:
        pool = list(parsed)
    else:
        pool = []
    if include_dangling:
        pool = pool + dangling

    keep_last_versions = (
        _keep_last_versions(parsed, keep_last) if keep_last is not None else {}
    )

    plan = PrunePlan()
    for image in pool:
        reason = _protection_reason(
            image, keep_last_versions, keep_versions, keep_active
        )
        if reason is None:
            plan.deleted.append(image)
        else:
            plan.protected.append((image, reason))
    return plan


def _protection_reason(
    image: ManagedImage,
    keep_last_versions: dict[str, set[str]],
    keep_versions: set[str],
    keep_active: bool,
) -> str | None:
    if keep_active and image.in_use:
        return "keep-active"
    if image.version is not None and image.version in keep_versions:
        return "keep"
    if (
        image.target is not None
        and image.version is not None
        and image.version in keep_last_versions.get(image.target, set())
    ):
        return "keep-last"
    return None
