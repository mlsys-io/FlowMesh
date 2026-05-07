"""Permission-checker hook.

Filters and gates access to every `ResourceType` in the permission
contract. The two methods cover the two read patterns:

- `accessible_ids` — bulk filter for list endpoints. Returns either `None`
  (no filter) or a `frozenset[str]` of resource ids the principal may see.
- `require` — point check for get / cancel / mutate endpoints. Raises
  to deny (FastAPI's HTTPException(403) is the documented choice).

Multiple checkers compose: `accessible_ids` returns `None` if any checker
returns `None`, otherwise the union of returned id sets; `require` requires
every checker to pass. With no checkers registered both helpers are no-ops.
"""

import logging
from typing import Protocol, runtime_checkable

from .types import PrincipalContext, ResourceAction, ResourceType


@runtime_checkable
class PermissionChecker(Protocol):
    name: str

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        action: ResourceAction,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        """Return the ids of `resource_type` the principal may `action`.

        Returns `None` to opt out of filtering, or a `frozenset[str]` of permitted ids
        (possibly empty).
        """
        ...

    async def require(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        resource_id: str | None,
        action: ResourceAction,
        logger: logging.Logger,
    ) -> None:
        """Raise if the principal may not `action` the resource.

        `resource_id=None` is a type-level / fleet-level check — "may the principal
        `action` resources of this type" — used at create-time before any id has been
        minted, and for `SYSTEM`-typed checks that have no associated id. Plugins should
        treat `None` distinctly from any concrete id; never compare against `""`.
        """
        ...
