import datetime


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def parse_iso_datetime(value: str | None) -> datetime.datetime | None:
    """Parse an ISO 8601 timestamp string. ``None`` / empty returns ``None``;
    invalid strings raise ``ValueError``. The ``Z`` suffix is normalized to
    ``+00:00`` so ``datetime.fromisoformat`` accepts it on Python <3.11."""
    if not value:
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
