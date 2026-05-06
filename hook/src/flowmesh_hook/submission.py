"""Submission-guard hook.

Runs before a workflow submission is parsed. Each guard self-filters by
principal (e.g. a balance guard short-circuits for non-billable orgs) and
raises an exception to block, returns None to allow.
"""

import logging
from typing import Protocol, runtime_checkable

from .types import PrincipalContext


@runtime_checkable
class SubmissionGuard(Protocol):
    name: str

    async def check(self, principal: PrincipalContext, logger: logging.Logger) -> None:
        """Reject the submission by raising, or allow."""
        ...
