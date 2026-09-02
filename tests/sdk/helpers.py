"""Reusable helpers for SDK tests."""

import enum
import os
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from flowmesh import ConfigNotFoundError
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .router_app import TEST_BASE_URL


def _is_truly_optional(field_info: FieldInfo) -> bool:
    """Return True if a field is genuinely optional from the API perspective.

    Server models use ``default_factory`` for internal construction (e.g.,
    ``default_factory=list``, ``default_factory=now_iso``).  These fields
    are *always* present in API responses — they are NOT optional from the
    SDK consumer's perspective.

    A field is truly optional only if its default is ``None``.
    """
    if field_info.is_required():
        return False
    if field_info.default is None:
        return True
    return False


def assert_fields_match(
    server_model: type[BaseModel],
    sdk_model: type[BaseModel],
    allow_sdk_extra: bool = True,
    skip_server_fields: set[str] | None = None,
) -> None:
    """Assert that SDK model fields are compatible with server model fields.

    For every visible server field (excluding ``exclude=True`` internals):
    - The SDK model must also have it.
    - If the server field is *truly* optional (default is None / API may omit),
      the SDK field must not be required.
    """
    server_fields = server_model.model_fields
    sdk_fields = sdk_model.model_fields

    excluded = {name for name, info in server_fields.items() if info.exclude}
    if skip_server_fields:
        excluded |= skip_server_fields
    server_visible = {k: v for k, v in server_fields.items() if k not in excluded}

    missing = set(server_visible) - set(sdk_fields)
    assert not missing, f"{sdk_model.__name__} missing server fields: {sorted(missing)}"

    if not allow_sdk_extra:
        extra = set(sdk_fields) - set(server_visible)
        assert (
            not extra
        ), f"{sdk_model.__name__} has extra fields not in server: {sorted(extra)}"

    for name in server_visible:
        server_truly_optional = _is_truly_optional(server_visible[name])
        sdk_req = sdk_fields[name].is_required()
        if sdk_req and server_truly_optional:
            pytest.fail(
                f"{sdk_model.__name__}.{name}: "
                f"optional in server API response (default=None) "
                f"but required in SDK"
            )


def assert_field_aliases_match(
    server_model: type[BaseModel],
    sdk_model: type[BaseModel],
) -> None:
    """Assert SDK and server field aliases agree for every shared field.

    A field's wire key is its alias; a divergence between the SDK and server
    alias (e.g. ``APIResult.response_json`` <-> ``json``) passes the
    name/optionality check yet breaks deserialization at the boundary.
    """
    server_fields = server_model.model_fields
    sdk_fields = sdk_model.model_fields
    for name, info in server_fields.items():
        if info.exclude or name not in sdk_fields:
            continue
        sdk_alias = sdk_fields[name].alias
        assert info.alias == sdk_alias, (
            f"{sdk_model.__name__}.{name}: alias mismatch "
            f"(server={info.alias!r}, sdk={sdk_alias!r})"
        )


def assert_extra_policy_matches(
    server_model: type[BaseModel],
    sdk_model: type[BaseModel],
) -> None:
    """Assert SDK and server models share the same ``extra`` policy.

    The discriminated union relies on concrete subclasses being
    ``extra="forbid"`` and the base staying permissive; a mismatch between the
    two mirrors would let one side silently accept payloads the other rejects.
    """
    server_extra = server_model.model_config.get("extra")
    sdk_extra = sdk_model.model_config.get("extra")
    assert server_extra == sdk_extra, (
        f"{sdk_model.__name__}: extra policy mismatch "
        f"(server={server_extra!r}, sdk={sdk_extra!r})"
    )


def assert_enum_members_match(
    server_enum: type[enum.Enum], sdk_enum: type[enum.Enum]
) -> None:
    """Assert that two StrEnum classes have identical member values."""
    server_values = sorted(m.value for m in server_enum)
    sdk_values = sorted(m.value for m in sdk_enum)
    assert server_values == sdk_values, (
        f"Enum mismatch between {server_enum.__name__} and {sdk_enum.__name__}:\n"
        f"  server: {server_values}\n"
        f"  sdk:  {sdk_values}"
    )


class SyncResponse:
    def __init__(
        self,
        lines: list[str] | None = None,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        url: str = f"{TEST_BASE_URL}/api/v1/tasks/t-1/logs/stream",
        json_body: Any = None,
        text_body: str | None = None,
    ) -> None:
        self._lines = lines or []
        self._chunks = chunks or []
        self.status_code = status_code
        self.url = url
        self._json_body = json_body
        self._text_body = text_body if text_body is not None else "body"
        self.content = self._text_body.encode()

    def iter_lines(self) -> list[str]:
        return list(self._lines)

    def iter_bytes(self, chunk_size: int = 0) -> list[bytes]:
        return list(self._chunks)

    def read(self) -> bytes:
        if self._chunks:
            self.content = b"".join(self._chunks)
        return self.content

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class AsyncResponse:
    def __init__(
        self,
        lines: list[str] | None = None,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        url: str = f"{TEST_BASE_URL}/api/v1/tasks/t-1/logs/stream",
        json_body: Any = None,
        text_body: str | None = None,
    ) -> None:
        self._lines = lines or []
        self._chunks = chunks or []
        self.status_code = status_code
        self.url = url
        self._json_body = json_body
        self._text_body = text_body if text_body is not None else "body"
        self.content = self._text_body.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_bytes(self, chunk_size: int = 0):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        if self._chunks:
            self.content = b"".join(self._chunks)
        return self.content

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class SyncStreamContext:
    def __init__(self, response: SyncResponse) -> None:
        self._response = response

    def __enter__(self) -> SyncResponse:
        return self._response

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class AsyncStreamContext:
    def __init__(self, response: AsyncResponse) -> None:
        self._response = response

    async def __aenter__(self) -> AsyncResponse:
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class SyncHTTP:
    def __init__(
        self,
        response: SyncResponse | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._exc = exc

    def stream(self, method: str, url: str, **kwargs: Any) -> SyncStreamContext:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return SyncStreamContext(self._response)

    def close(self) -> None:
        return None


class AsyncHTTP:
    def __init__(
        self,
        response: AsyncResponse | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._exc = exc

    def stream(self, method: str, url: str, **kwargs: Any) -> AsyncStreamContext:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return AsyncStreamContext(self._response)

    async def aclose(self) -> None:
        return None


@contextmanager
def clear_env_and_config_file():
    with patch.dict(os.environ, {}, clear=True):
        for key in ("FLOWMESH_BASE_URL", "FLOWMESH_API_KEY"):
            os.environ.pop(key, None)
        with patch(
            "flowmesh._base_client.FlowMeshConfig.from_file",
            side_effect=ConfigNotFoundError("no config"),
        ):
            yield
