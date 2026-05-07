"""`UsageSink` example: append rows to an in-memory ledger and log a summary."""

import logging
from decimal import Decimal

from flowmesh_hook import UsageRow

from . import state


class SimpleUsageSink:
    name = "simple_plugin.usage"

    async def emit(self, rows: list[UsageRow], logger: logging.Logger) -> None:
        if not rows:
            return
        state.USAGE_LEDGER.extend(rows)
        total_cost = sum((row["cost"] for row in rows), Decimal("0"))
        logger.info(
            "%s: appended %d row(s), total_cost=%s, ledger_size=%d",
            self.name,
            len(rows),
            total_cost,
            len(state.USAGE_LEDGER),
        )
