"""Classify transient HTTP statuses and parse retry delays."""

import email.utils
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol


class RetryResponse(Protocol):
    """The HTTP response fields needed to decide whether and when to retry."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


def is_retryable_status(status_code: int, *, idempotent: bool = True) -> bool:
    """Whether an HTTP status is transient and safe to retry."""
    if not idempotent:
        return False
    return status_code >= 500 or status_code in (408, 429)


def retry_after_seconds(response: RetryResponse) -> float | None:
    """Parse Retry-After as a non-negative delay or an HTTP date."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
