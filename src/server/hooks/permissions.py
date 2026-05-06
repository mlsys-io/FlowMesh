"""Permission-checker hook.

Filters and gates access to workflow / task / worker / node resources. The
two methods cover the two read patterns:

- `accessible_ids` — bulk filter for list endpoints. Returns either the
  literal `"all"` (no filter) or a `frozenset[str]` of resource ids the
  principal may see.
- `require` — point check for get / cancel / mutate endpoints. Raises
  `HTTPException(403)` to deny.

Multiple checkers compose: `accessible_ids` returns `"all"` if any checker
returns `"all"`, otherwise the union of returned id sets; `require` requires
every checker to pass. With no checkers registered both helpers are no-ops,
matching OSS's open-by-default behaviour.
"""

import logging
from typing import Literal, Protocol, runtime_checkable

from ..auth.security import PrincipalContext

AccessibleIds = Literal["all"] | frozenset[str]


@runtime_checkable
class PermissionChecker(Protocol):
    name: str

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        resource_type: str,
        action: str,
        logger: logging.Logger,
    ) -> AccessibleIds:
        """Return the ids of `resource_type` the principal may `action`.

        Returns `"all"` to opt out of filtering, or a `frozenset[str]` of
        permitted ids (possibly empty).
        """
        ...

    async def require(
        self,
        principal: PrincipalContext,
        resource_type: str,
        resource_id: str,
        action: str,
        logger: logging.Logger,
    ) -> None:
        """Raise `HTTPException(403)` if the principal may not `action` the resource."""
        ...
