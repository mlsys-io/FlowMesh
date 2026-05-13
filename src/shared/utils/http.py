"""Shared HTTP helpers: auth headers and a small session wrapper."""

import os


def auth_headers(token: str | None = None) -> dict[str, str]:
    """Return `{Authorization: Bearer <token>}` if a token is available.

    Reads `FLOWMESH_API_KEY` from the environment when `token` is None.
    """
    if token is None:
        token = os.getenv("FLOWMESH_API_KEY", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def add_auth_headers(headers: dict[str, str], token: str | None = None) -> None:
    """Insert auth headers into `headers` unless `Authorization` is already set."""
    if not any(k.lower() == "authorization" for k in headers):
        headers.update(auth_headers(token))
