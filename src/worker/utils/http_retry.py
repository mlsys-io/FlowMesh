"""Classify transient HTTP statuses and compute retry delays."""

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


def is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
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


def backoff_delay(
    attempt: int,
    base_sec: float,
    max_sec: float,
    response: RetryResponse | None = None,
) -> float:
    """The wait before retry ``attempt``: the response's Retry-After when it has
    one, else ``base_sec`` doubled per retry, capped at ``max_sec`` either way."""
    if response is not None:
        retry_after = retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, max_sec)
    return min(base_sec * (2 ** (attempt - 1)), max_sec)
