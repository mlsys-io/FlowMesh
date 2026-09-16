"""Redaction of API credentials before a task record is serialized.

The in-memory ``TaskRecord`` keeps the real credential so dispatch works; the
serializer applies this module so that every dump to Redis is redacted.

``raw_yaml`` is re-emitted via ``yaml.safe_dump``, which does not preserve
comments, key order or original formatting. That is an accepted cost: the field
is a stored record and is never re-parsed, so losing formatting is fine. If the
YAML cannot be parsed, the whole field is redacted rather than storing text
that might contain a key.
"""

from typing import Any

import yaml

REDACTED = "[REDACTED]"

# Whole-key matches; a bare substring test would over-redact (e.g. "monkey").
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "token",
        "api-key",
        "api_key",
        "apikey",
        "secret",
        "access_token",
        "bearer",
        "x-api-key",
    }
)
_SENSITIVE_SUFFIXES = ("_key", "-key", "_token", "-token")


def _is_sensitive_key(name: str) -> bool:
    lowered = name.lower()
    if lowered in _SENSITIVE_KEYS:
        return True
    return any(lowered.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _redact_value(value: Any) -> Any:
    """Recursively redact credential values by key name at any depth."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_sensitive_key(str(key)) else _redact_value(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_api(api: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of an api spec with credential values replaced.

    Only the five credential-bearing locations are touched; the rest of the
    spec is returned unchanged. The original mapping is never mutated.
    """
    if not isinstance(api, dict):
        return api
    redacted = dict(api)
    for field in ("headers", "params", "body", "json", "data"):
        value = redacted.get(field)
        if isinstance(value, (dict, list)):
            redacted[field] = _redact_value(value)
    return redacted


def redact_raw_yaml(raw_yaml: str) -> str:
    """Redact credential values from the original workflow YAML text.

    The YAML is parsed and redacted with the same recursive key rule used for
    the parsed spec, then re-emitted. If parsing fails, the whole field is
    replaced with a marker rather than storing un-analysed text.
    """
    try:
        tree = yaml.safe_load(raw_yaml)
    except yaml.YAMLError:
        return REDACTED
    if tree is None:
        return raw_yaml
    return yaml.safe_dump(_redact_value(tree), sort_keys=False)
