"""Format of an external worker token: `<alias>.<hex HMAC-SHA256 of alias>`.

Shared by the supervisor, which mints and verifies tokens, and the worker, which
reads its own alias from one.
"""

import string

EXTERNAL_ALIAS_SEP = "."
_DIGEST_LEN = 64
_HEX_DIGITS = frozenset(string.hexdigits.lower())


def split_external_token(token: str) -> tuple[str, str] | None:
    """Split a token into `(alias, digest)` if it has the external token shape.

    An alias may itself contain dots, so the split is on the last one.
    """
    alias, sep, digest = token.rpartition(EXTERNAL_ALIAS_SEP)
    if not sep or not alias or len(digest) != _DIGEST_LEN:
        return None
    if not _HEX_DIGITS.issuperset(digest):
        return None
    return alias, digest


def external_token_alias(token: str) -> str | None:
    """Alias prefix of an external worker token, unverified."""
    parts = split_external_token(token)
    return parts[0] if parts is not None else None


__all__ = ["EXTERNAL_ALIAS_SEP", "external_token_alias", "split_external_token"]
