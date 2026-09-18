"""Redacts credential-shaped fields from an object before it is serialized.

Applies a key-based rule: any header, parameter, or nested field whose name
looks like a credential (see ``is_sensitive_key``) has its value replaced with
a fixed marker. Callers keep an unredacted copy for in-process use and apply
this module only at the serialization boundary.

Raw YAML text is redacted by parsing and re-emitting it via
``yaml.safe_dump``, which does not preserve comments, key order or original
formatting. That is an accepted cost for a value that is stored and never
re-parsed. If the YAML cannot be parsed, the whole field is redacted rather
than storing text that might contain a key.
"""

from typing import Any

import yaml

REDACTED = "[REDACTED]"

# Whole-key matches; a bare substring test would over-redact (e.g. "monkey").
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "authorizedkeys",
        "cert_data",
        "certificate",
        "connection_string",
        "token",
        "api-key",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "access_token",
        "bearer",
        "x-api-key",
    }
)
_SENSITIVE_SUFFIXES = (
    "_key",
    "-key",
    "_password",
    "-password",
    "_secret",
    "-secret",
    "_token",
    "-token",
)
_SENSITIVE_COMBINATIONS = frozenset(
    {
        "access_key",
        "api_key",
        "auth_key",
        "credential_key",
        "private_key",
        "secret_key",
    }
)


def is_sensitive_key(name: str) -> bool:
    """Whether a header/param name carries a credential value."""
    lowered = name.lower()
    if lowered in _SENSITIVE_KEYS:
        return True
    if any(lowered.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES):
        return True
    normalized = lowered.replace("-", "_")
    return any(combination in normalized for combination in _SENSITIVE_COMBINATIONS)


def redact_value(value: Any) -> Any:
    """Recursively redact credential values by key name at any depth."""
    if isinstance(value, dict):
        return {
            key: (
                [REDACTED]
                if is_sensitive_key(str(key)) and isinstance(val, list)
                else REDACTED if is_sensitive_key(str(key)) else redact_value(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def redact_credential(value: Any) -> Any:
    """Replace a known scalar credential while preserving an omitted value."""
    if value is None:
        return None
    if isinstance(value, list):
        return [REDACTED]
    return REDACTED


def is_redacted(value: Any) -> bool:
    """Whether a scalar value is the redaction marker."""
    return value == REDACTED


def contains_redacted(value: Any) -> bool:
    """Whether a value contains a redacted placeholder at any depth."""
    if value == REDACTED:
        return True
    if isinstance(value, dict):
        return any(contains_redacted(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_redacted(item) for item in value)
    return False


def contains_redacted_credential(value: Any) -> bool:
    """Whether a redacted placeholder occurs under a credential-shaped key."""
    if isinstance(value, dict):
        return any(
            (
                contains_redacted(item)
                if is_sensitive_key(str(key))
                else contains_redacted_credential(item)
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_redacted_credential(item) for item in value)
    return False


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
    return yaml.safe_dump(redact_value(tree), sort_keys=False)
