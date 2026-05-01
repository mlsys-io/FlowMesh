import datetime


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def parse_iso_datetime(value: str | None) -> datetime.datetime | None:
    """Parse ISO 8601 → ``datetime``; ``None`` if empty, raises on malformed."""
    if not value:
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
