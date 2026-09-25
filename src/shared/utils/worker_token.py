"""Format of an external worker token: `<name>.<hex HMAC-SHA256 of name>`.

Shared by the supervisor, which mints and verifies tokens, and the worker, which
reads its own name from one.
"""

import string

EXTERNAL_NAME_SEP = "."
_DIGEST_LEN = 64
_HEX_DIGITS = frozenset(string.hexdigits.lower())


def split_external_token(token: str) -> tuple[str, str] | None:
    """Split a token into `(name, digest)` if it has the external token shape.

    A name may itself contain dots, so the split is on the last one.
    """
    name, sep, digest = token.rpartition(EXTERNAL_NAME_SEP)
    if not sep or not name or len(digest) != _DIGEST_LEN:
        return None
    if not _HEX_DIGITS.issuperset(digest):
        return None
    return name, digest


def external_token_name(token: str) -> str | None:
    """Name prefix of an external worker token, unverified."""
    parts = split_external_token(token)
    return parts[0] if parts is not None else None


__all__ = ["EXTERNAL_NAME_SEP", "external_token_name", "split_external_token"]
