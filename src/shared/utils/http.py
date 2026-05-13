"""Shared HTTP helpers: auth headers and a small session wrapper."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import requests


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


class HttpSession:
    def __init__(
        self, base_url: str, token: str, version_prefix: str = "/api/v1"
    ) -> None:
        self.base_url = base_url
        self._version_prefix = version_prefix
        self._auth_headers = auth_headers(token)
        self._session = requests.Session()
        if self._auth_headers:
            self._session.headers.update(self._auth_headers)

    def _make_url(self, path: str, version_prefix: bool) -> str:
        url = self.base_url.rstrip("/")
        if version_prefix:
            url += self._version_prefix
        url += "/" + path.lstrip("/")
        return url

    def _add_auth_headers(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs["headers"] = self._auth_headers | kwargs.get("headers", {})
        return kwargs

    def request(
        self, method: str, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        return self._session.request(method, url, **kwargs)

    @asynccontextmanager
    async def arequest(
        self, method: str, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        url = self._make_url(path, version_prefix)
        kwargs = self._add_auth_headers(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                yield response

    def get(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        return self._session.get(url, **kwargs)

    @asynccontextmanager
    async def aget(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        url = self._make_url(path, version_prefix)
        kwargs = self._add_auth_headers(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **kwargs) as response:
                yield response

    def post(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        return self._session.post(url, **kwargs)

    @asynccontextmanager
    async def apost(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        url = self._make_url(path, version_prefix)
        kwargs = self._add_auth_headers(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, **kwargs) as response:
                yield response

    def put(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        return self._session.put(url, **kwargs)

    @asynccontextmanager
    async def aput(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        url = self._make_url(path, version_prefix)
        kwargs = self._add_auth_headers(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.put(url, **kwargs) as response:
                yield response

    def delete(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        return self._session.delete(url, **kwargs)

    @asynccontextmanager
    async def adelete(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        url = self._make_url(path, version_prefix)
        kwargs = self._add_auth_headers(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, **kwargs) as response:
                yield response
