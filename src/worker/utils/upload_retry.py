"""Retry worker uploads across transient failures.

FlowMesh's own upload endpoints (result, artifact and trace files, the system
delivery bundle) overwrite atomically, so re-sending after an ambiguous failure
(a read timeout on a request the server may already have stored) is safe; those
uploads are idempotent. An external destination gets a retry only when the
request failed before connecting to it.
"""

import email.utils
import logging
import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol

import httpx
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from shared.utils.parsing import parse_float_env, parse_int_env

_BACKOFF_MAX_SEC = 30.0


class _Response(Protocol):
    """The part of a ``requests`` or ``httpx`` response the retry loop reads."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


def upload_retries() -> int:
    """Retries after the first attempt (``WORKER_UPLOAD_RETRIES``, default 5)."""
    return max(0, parse_int_env("WORKER_UPLOAD_RETRIES", 5))


def upload_backoff_sec() -> float:
    """First backoff, doubled per retry (``WORKER_UPLOAD_BACKOFF_SEC``, default 2)."""
    return max(0.0, parse_float_env("WORKER_UPLOAD_BACKOFF_SEC", 2.0))


def is_retryable_status(status_code: int, *, idempotent: bool = True) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
    if not idempotent:
        return False
    return status_code >= 500 or status_code in (408, 429)


def never_reached_server(exc: BaseException) -> bool:
    """A failure to open the connection: refused, unresolvable, connect timeout."""
    if isinstance(
        exc, (requests.ConnectTimeout, httpx.ConnectError, httpx.ConnectTimeout)
    ):
        return True
    if isinstance(exc, requests.ConnectionError) and exc.args:
        cause = exc.args[0]
        return isinstance(cause, MaxRetryError) and isinstance(
            cause.reason, NewConnectionError
        )
    return False


def is_retryable_error(exc: BaseException, *, idempotent: bool = True) -> bool:
    """Connection-level failures; for a non-idempotent upload, only those that
    never reached the server."""
    if not idempotent:
        return never_reached_server(exc)
    return isinstance(
        exc, (requests.ConnectionError, requests.Timeout, httpx.TransportError)
    )


def _retry_after_seconds(response: _Response) -> float | None:
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
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _backoff(attempt: int, response: _Response | None) -> float:
    if response is not None:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return retry_after
    return min(upload_backoff_sec() * (2 ** (attempt - 1)), _BACKOFF_MAX_SEC)


def send_with_retries[R: _Response](
    send: Callable[[], R],
    *,
    what: str,
    idempotent: bool = True,
    logger: logging.Logger | None = None,
    sleep: Callable[[float], None] | None = None,
) -> R:
    """Send a rebuilt request with bounded retries.

    ``send`` must build the whole request on every call (re-open files, rewind
    streams). ``idempotent=False`` retries only failures before connecting.
    Return the final response, including a transient status after exhaustion.
    Raise the final exception; non-retryable exceptions propagate immediately.
    """
    retries = upload_retries()
    wait = time.sleep if sleep is None else sleep
    attempt = 0
    while True:
        try:
            response = send()
        except Exception as exc:
            if not is_retryable_error(exc, idempotent=idempotent) or attempt >= retries:
                raise
            attempt += 1
            delay = _backoff(attempt, None)
            if logger:
                logger.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    what,
                    attempt,
                    retries + 1,
                    exc,
                    delay,
                )
            wait(delay)
            continue
        if (
            not is_retryable_status(response.status_code, idempotent=idempotent)
            or attempt >= retries
        ):
            return response
        attempt += 1
        delay = _backoff(attempt, response)
        if logger:
            logger.warning(
                "%s returned %s (attempt %d/%d); retrying in %.1fs",
                what,
                response.status_code,
                attempt,
                retries + 1,
                delay,
            )
        wait(delay)
