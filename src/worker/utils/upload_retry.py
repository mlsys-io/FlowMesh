"""Retry worker uploads across transient failures.

FlowMesh's own upload endpoints (result, artifact and trace files, the system
delivery bundle) overwrite atomically, so re-sending after an ambiguous failure
(a read timeout on a request the server may already have stored) is safe; those
uploads are idempotent. An external destination gets a retry only when the
request failed before connecting to it.
"""

import logging
import ssl
import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from shared.utils.parsing import parse_float_env, parse_int_env

from .http_retry import RetryResponse, backoff_delay, is_retryable_status

_BACKOFF_MAX_SEC = 60.0
_PERMANENT_ERRORS = (httpx.UnsupportedProtocol, httpx.LocalProtocolError)

# Origins whose last upload ended on a connection failure. Uploads to them make a
# single attempt until one gets a response, so a down server costs one retry
# schedule in total.
_unreachable_origins: set[str] = set()


def upload_retries() -> int:
    """Retries after the first attempt (``WORKER_UPLOAD_RETRIES``, default 5)."""
    return max(0, parse_int_env("WORKER_UPLOAD_RETRIES", 5))


def upload_backoff_sec() -> float:
    """First backoff, doubled per retry (``WORKER_UPLOAD_BACKOFF_SEC``, default 2)."""
    return max(0.0, parse_float_env("WORKER_UPLOAD_BACKOFF_SEC", 2.0))


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


def _failed_certificate_verification(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def is_retryable_error(exc: BaseException, *, idempotent: bool) -> bool:
    """Transient connection-level failures; for a non-idempotent upload, only
    those that never reached the server."""
    if isinstance(exc, _PERMANENT_ERRORS) or _failed_certificate_verification(exc):
        return False
    if not idempotent:
        return never_reached_server(exc)
    return isinstance(
        exc, (requests.ConnectionError, requests.Timeout, httpx.TransportError)
    )


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _retry_delay(
    attempt: int,
    response: RetryResponse | None,
    started: float,
    deadline_sec: float | None,
) -> float | None:
    """The wait before retry ``attempt``, or None when it would pass the deadline."""
    delay = backoff_delay(attempt, upload_backoff_sec(), _BACKOFF_MAX_SEC, response)
    if deadline_sec is not None and time.monotonic() - started + delay > deadline_sec:
        return None
    return delay


def send_with_retries[R: RetryResponse](
    send: Callable[[], R],
    *,
    url: str,
    what: str,
    idempotent: bool,
    deadline_sec: float | None = None,
    logger: logging.Logger | None = None,
) -> R:
    """Send a rebuilt request to ``url`` with bounded retries.

    ``send`` must build the whole request on every call (re-open files, rewind
    streams). ``idempotent=False`` retries only failures before connecting.
    Retries stop once the next wait would end more than ``deadline_sec`` after
    the first attempt began. After a call to an origin ends on a connection
    failure, calls to it make a single attempt until one gets a response.
    Return the final response, including a transient status when retries end.
    Raise the final exception; non-retryable exceptions propagate immediately.
    """
    origin = _origin(url)
    retries = 0 if origin in _unreachable_origins else upload_retries()
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            response = send()
        except Exception as exc:
            if not is_retryable_error(exc, idempotent=idempotent):
                raise
            delay = (
                _retry_delay(attempt + 1, None, started, deadline_sec)
                if attempt < retries
                else None
            )
            if delay is None:
                _unreachable_origins.add(origin)
                raise
            attempt += 1
            if logger:
                logger.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    what,
                    attempt,
                    retries,
                    exc,
                    delay,
                )
            time.sleep(delay)
            continue
        _unreachable_origins.discard(origin)
        if (
            not idempotent
            or not is_retryable_status(response.status_code)
            or attempt >= retries
        ):
            return response
        delay = _retry_delay(attempt + 1, response, started, deadline_sec)
        if delay is None:
            return response
        attempt += 1
        if logger:
            logger.warning(
                "%s returned %s (attempt %d/%d); retrying in %.1fs",
                what,
                response.status_code,
                attempt,
                retries,
                delay,
            )
        time.sleep(delay)
