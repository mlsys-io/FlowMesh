"""Usage-sink hook.

Fan-out for per-task usage rows after a task completes. Each sink decides
which rows it consumes and how to deliver them. Sink failures are isolated by
the caller — they must not break the dispatch path.
"""

import logging
from typing import Protocol, runtime_checkable

from .types import UsageRow


@runtime_checkable
class UsageSink(Protocol):
    name: str

    async def emit(self, rows: list[UsageRow], logger: logging.Logger) -> None:
        """Deliver per-task usage rows to a downstream sink."""
        ...
