"""In-memory state for `simple_plugin`.

All seed data and mutable stores live here so each hook module can read and
write a single shared source of truth. Names are public (no leading
underscore) because the README documents them as the knobs you tweak when
poking at the plugin.
"""

from flowmesh_hook import UsageRow
from lumid_hooks import PrincipalContext

TOKENS: dict[str, PrincipalContext] = {
    "demo-admin-token": PrincipalContext(
        principal_id="alice",
        org_id="demo",
        external_id="alice@example.com",
        principal_type="admin",
        scopes=["admin"],
    ),
    "demo-user-token": PrincipalContext(
        principal_id="bob",
        org_id="demo",
        external_id="bob@example.com",
        principal_type="user",
        scopes=["user"],
    ),
}

# Add a principal_id here to make `SimpleSubmissionGuard` reject its
# workflows with HTTP 403.
BLOCKED_PRINCIPALS: set[str] = set()

# Populated by `SimpleResourceRegistrar.register`; read by
# `SimplePermissionChecker` to gate non-admin access.
OWNERSHIP: dict[tuple[str, str], str] = {}

# Appended by `SimpleUsageSink.emit` — kept around for inspection so users
# can confirm tasks complete and rows fan out as expected.
USAGE_LEDGER: list[UsageRow] = []
