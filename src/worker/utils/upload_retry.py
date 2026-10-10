"""Retry worker uploads across transient failures.

FlowMesh's own upload endpoints (result, artifact and trace files, the system
delivery bundle) overwrite atomically, so re-sending after an ambiguous failure
(a read timeout on a request the server may already have stored) is safe; those
uploads are idempotent. An external destination gets a retry only when the
request failed before connecting to it.
"""

import logging
import time
from collections.abc import Callable

import httpx
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from shared.utils.parsing import parse_float_env, parse_int_env

from .http_retry import RetryResponse, is_retryable_status, retry_after_seconds

UPLOAD_RETRIES = max(0, parse_int_env("WORKER_UPLOAD_RETRIES", 5))
UPLOAD_BACKOFF_SEC = max(0.0, parse_float_env("WORKER_UPLOAD_BACKOFF_SEC", 2.0))
_BACKOFF_MAX_SEC = 30.0


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


def _backoff(attempt: int, response: RetryResponse | None) -> float:
    if response is not None:
        retry_after = retry_after_seconds(response)
        if retry_after is not None:
            return retry_after
    return min(UPLOAD_BACKOFF_SEC * (2 ** (attempt - 1)), _BACKOFF_MAX_SEC)


def send_with_retries[R: RetryResponse](
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
    retries = UPLOAD_RETRIES
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
