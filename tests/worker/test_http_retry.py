import email.utils
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from worker.utils.http_retry import (
    backoff_delay,
    is_retryable_status,
    retry_after_seconds,
)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_transient_statuses_are_retryable(status: int) -> None:
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 413])
def test_other_statuses_are_not_retryable(status: int) -> None:
    assert not is_retryable_status(status)


@pytest.mark.parametrize(
    "value",
    [
        "nan",
        "inf",
        "-1",
        "not-a-date",
        "Fri, 31 Dec 99999999999999999999 23:59:59 GMT",
    ],
)
def test_invalid_retry_after_returns_no_delay(value: str) -> None:
    response = httpx.Response(503, headers={"Retry-After": value})
    assert retry_after_seconds(response) is None


def test_missing_retry_after_returns_no_delay() -> None:
    assert retry_after_seconds(httpx.Response(503)) is None


@pytest.mark.parametrize("delay", [0, 7, 300])
def test_retry_after_seconds_are_not_capped(delay: int) -> None:
    response = httpx.Response(503, headers={"Retry-After": str(delay)})
    assert retry_after_seconds(response) == delay


@pytest.mark.parametrize("aware", [False, True])
def test_retry_after_http_date(aware: bool) -> None:
    retry_at = datetime.now(UTC) + timedelta(seconds=30)
    if not aware:
        retry_at = retry_at.replace(tzinfo=None)
    response = httpx.Response(
        503, headers={"Retry-After": email.utils.format_datetime(retry_at)}
    )
    assert retry_after_seconds(response) == pytest.approx(30, abs=1)


def test_past_retry_after_http_date_returns_zero() -> None:
    retry_at = datetime.now(UTC) - timedelta(seconds=30)
    response = httpx.Response(
        503, headers={"Retry-After": email.utils.format_datetime(retry_at)}
    )
    assert retry_after_seconds(response) == 0


def test_backoff_doubles_up_to_the_cap() -> None:
    assert [backoff_delay(attempt, 2.0, 60.0) for attempt in range(1, 7)] == [
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        60.0,
    ]


@pytest.mark.parametrize(("header", "delay"), [("0", 0.0), ("35", 35.0), ("999", 60.0)])
def test_retry_after_replaces_backoff_under_the_cap(header: str, delay: float) -> None:
    response = httpx.Response(503, headers={"Retry-After": header})

    assert backoff_delay(3, 2.0, 60.0, response) == delay
