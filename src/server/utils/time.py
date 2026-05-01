import time

from shared.utils.time import now_iso, parse_iso_datetime


def parse_iso_ts(value: str | None) -> float:
    """Best-effort ISO 8601 → Unix timestamp; falls back to ``time.time()``
    on missing or malformed input. The fallback is intentional: callers feed
    this to telemetry / heartbeat fields that need a number even when the
    upstream string is broken.
    """
    try:
        dt = parse_iso_datetime(value)
    except ValueError:
        return time.time()
    return dt.timestamp() if dt is not None else time.time()


__all__ = ["now_iso", "parse_iso_ts"]
