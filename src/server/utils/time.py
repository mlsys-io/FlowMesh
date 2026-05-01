import time

from shared.utils.time import now_iso, parse_iso_datetime


def parse_iso_ts(value: str | None) -> float:
    """ISO 8601 → Unix timestamp; ``time.time()`` on missing / malformed."""
    try:
        dt = parse_iso_datetime(value)
    except ValueError:
        return time.time()
    return dt.timestamp() if dt else time.time()


__all__ = ["now_iso", "parse_iso_ts"]
