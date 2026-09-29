import email.utils
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx

from shared.schemas.result import APIGroupItem, APIItem, APIResult
from shared.tasks.specs import ApiSpecStrict
from shared.tasks.specs.misc import ApiConfig, ApiResponseConfig
from shared.tasks.task_type import TaskType
from shared.utils.redact import is_credential_key

from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    TaskCancelledError,
)
from .mixins.data import DataMixin

logger = logging.getLogger(__name__)

# Cache key: (base_url, timeout_seconds, verify_tls, follow_redirects, concurrency)
_ClientKey = tuple[str, float, bool, bool, int]

# Worker-side per-row slot; server-side stage references are ${...}.
_PROMPT_PLACEHOLDER = "{{prompt}}"

# Base delay between retry attempts, doubled each retry.
_RETRY_BACKOFF_SEC = 1.0
# Upper bound on any single retry wait.
_RETRY_BACKOFF_MAX_SEC = 60.0


def _is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
    return status_code >= 500 or status_code in (408, 429)


class _RequestFailed(Exception):
    """A request that exhausted its retries, carrying the attempts used."""

    def __init__(self, error: Exception, attempts: int) -> None:
        super().__init__(str(error))
        self.error = error
        self.attempts = attempts


def _fmt(value: Any) -> str:
    """Render a log field, using ``-`` for a missing value."""
    return "-" if value is None else str(value)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of a non-empty list of latencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _chat_completion_stats(body: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Extract token and finish-reason fields from a chat-completion body.

    Returns ``(prompt_tokens, completion_tokens, reasoning_tokens,
    finish_reason, backend)`` with ``None`` for any missing field. Never
    raises on an unexpected body shape.
    """
    if not isinstance(body, dict):
        return None, None, None, None, None
    usage = body.get("usage")
    prompt_tokens = completion_tokens = reasoning_tokens = None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens")
    choices = body.get("choices")
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
    backend = body.get("provider")
    if backend is None:
        backend = body.get("system_fingerprint")
    return (
        _as_token_count(prompt_tokens),
        _as_token_count(completion_tokens),
        _as_token_count(reasoning_tokens),
        finish_reason,
        backend if isinstance(backend, str) else None,
    )


def _as_token_count(value: Any) -> int | None:
    """Coerce a token field to an int, skipping non-numeric values (including
    bools) so malformed telemetry never changes a request's outcome."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class APIExecutor(DataMixin, Executor):
    """Performs HTTP requests defined by task YAML.

    ``spec.data`` is required and yields one row per request: each row's prompt
    is substituted for the ``{{prompt}}`` placeholder in the request body and
    one request is issued per row, returned row-aligned in ``APIResult.items``.
    A single request is a one-row ``spec.data``.

    Defaults to the Nebula endpoint via ``NEBULA_API_BASE_URL`` and authenticates
    with ``NEBULA_API_TOKEN``. ``spec.api.url`` overrides the endpoint and
    ``spec.api.headers`` may supply a credential header (``Authorization``,
    ``X-API-Key``, etc.) directly. A custom ``spec.api.url`` may be
    unauthenticated; the Nebula token is never sent to an endpoint the caller
    chose.

    Parallel row requests and the HTTP connection pool are capped at
    ``_MAX_CONCURRENCY``; the pool is sized to the effective concurrency so
    parallel requests never queue on connections.
    """

    name = "api"
    supported_task_types = frozenset({TaskType.API})

    # ---- Class-level connection pool (shared across all instances) ----
    _clients: ClassVar[dict[_ClientKey, httpx.Client]] = {}
    _clients_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()
        self._cancel_lock = threading.Lock()
        self._active_task_id: str | None = None
        self._pending_cancelled_ids: set[str] = set()

    def cancel(self, task_id: str) -> None:
        """Signal the executor to abort the current request and any retries."""
        with self._cancel_lock:
            if self._active_task_id is None:
                # No run in flight: record the id so a pre-start cancel lands.
                self._pending_cancelled_ids.add(task_id)
                self._cancel_event.set()
            elif self._active_task_id == task_id:
                self._cancel_event.set()
            # A cancellation for a different task than the active one is ignored.

    @classmethod
    def _base_url(cls, url: str) -> str:
        """Extract scheme + host + port from a URL for pool keying."""
        parsed = httpx.URL(url)
        # httpx.URL exposes .scheme, .host, .port; rebuild origin string
        port = parsed.port
        if port is None:
            return f"{parsed.scheme}://{parsed.host}"
        return f"{parsed.scheme}://{parsed.host}:{port}"

    @classmethod
    def _get_client(
        cls,
        base_url: str,
        timeout: httpx.Timeout,
        verify_tls: bool,
        follow_redirects: bool,
        concurrency: int,
    ) -> httpx.Client:
        """Return a cached client or create one for the given parameters.

        The pool is sized to ``concurrency`` so parallel row requests never
        queue on connections."""
        timeout_sec = timeout.connect  # all four fields are set to same value
        if timeout_sec is None:
            timeout_sec = 0.0
        key: _ClientKey = (
            base_url,
            float(timeout_sec),
            verify_tls,
            follow_redirects,
            concurrency,
        )
        with cls._clients_lock:
            client = cls._clients.get(key)
            if client is not None and not client.is_closed:
                return client
            client = httpx.Client(
                timeout=timeout,
                verify=verify_tls,
                follow_redirects=follow_redirects,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
            )
            cls._clients[key] = client
            logger.debug(
                "Created new HTTP client for %s (verify=%s, timeout=%s, "
                "concurrency=%s)",
                base_url,
                verify_tls,
                timeout_sec,
                concurrency,
            )
            return client

    def _request_with_retries(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        headers: dict[str, Any],
        params: dict[str, Any] | None,
        request_kwargs: dict[str, Any],
        retries: int,
        failed: threading.Event,
    ) -> tuple[httpx.Response, int]:
        """Issue the request, retrying transient failures up to ``retries`` times.

        A retryable failure is a connection error or a transient HTTP status
        (5xx, 408, 429). Non-retryable failures, a cancelled task, or a row
        already failed elsewhere stop the loop immediately. The final attempt's
        failure propagates to the caller. Returns the response and the number
        of attempts used.
        """
        attempt = 0
        while True:
            if self._cancel_event.is_set():
                raise TaskCancelledError("API request cancelled")
            if failed.is_set():
                raise ExecutionError("API task failed on an earlier row")
            try:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    **request_kwargs,
                )
            except httpx.RequestError as exc:
                if attempt >= retries:
                    raise _RequestFailed(exc, attempt + 1) from None
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "API request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    retries,
                    exc,
                    delay,
                )
                self._wait_for_backoff(delay, failed)
                continue
            if resp.is_error and _is_retryable_status(resp.status_code):
                if attempt >= retries:
                    return resp, attempt + 1
                attempt += 1
                delay = self._backoff_delay(attempt, resp)
                logger.warning(
                    "API request returned %s (attempt %d/%d); retrying in %.1fs",
                    resp.status_code,
                    attempt,
                    retries,
                    delay,
                )
                self._wait_for_backoff(delay, failed)
                continue
            return resp, attempt + 1

    def _backoff_delay(self, attempt: int, resp: httpx.Response | None = None) -> float:
        """Return the wait before the next attempt, honouring Retry-After."""
        if resp is not None:
            retry_after = self._retry_after_seconds(resp)
            if retry_after is not None:
                return min(retry_after, _RETRY_BACKOFF_MAX_SEC)
        return min(_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), _RETRY_BACKOFF_MAX_SEC)

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response) -> float | None:
        """Parse the Retry-After header as seconds or an HTTP date."""
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            pass
        else:
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    def _wait_for_backoff(self, delay: float, failed: threading.Event) -> None:
        """Wait out the retry backoff, aborting early if the task is cancelled
        or another row has already failed."""
        deadline = time.monotonic() + delay
        while True:
            if self._cancel_event.is_set():
                raise TaskCancelledError("API request cancelled")
            if failed.is_set():
                raise ExecutionError("API task failed on an earlier row")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cancel_event.wait(min(remaining, 0.1))

    @classmethod
    def close_all_clients(cls) -> None:
        """Close and discard all cached HTTP clients."""
        with cls._clients_lock:
            for key, client in cls._clients.items():
                try:
                    client.close()
                except Exception:
                    logger.debug("Error closing HTTP client for %s", key[0])
            cls._clients.clear()
            logger.debug("All cached HTTP clients closed")

    def cleanup_after_run(self) -> None:
        """Close the connection pool when the runner deactivates this executor."""
        self.close_all_clients()

    @staticmethod
    def _prompt_to_str(prompt: Any) -> str:
        """Render a row's prompt as a string for body substitution."""
        if isinstance(prompt, str):
            return prompt
        return json.dumps(prompt)

    @classmethod
    def _substitute_prompt(cls, value: Any, prompt: Any) -> Any:
        """Replace ``{{prompt}}`` in the request body with a row's prompt.

        A value that is exactly ``{{prompt}}`` becomes the prompt object
        itself, so a chat-message list stays a list of ``{role, content}``
        dicts; a placeholder embedded in a longer string is rendered as text.
        """
        if isinstance(value, str):
            if value == _PROMPT_PLACEHOLDER:
                return prompt
            return value.replace(_PROMPT_PLACEHOLDER, cls._prompt_to_str(prompt))
        if isinstance(value, dict):
            return {k: cls._substitute_prompt(v, prompt) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._substitute_prompt(v, prompt) for v in value]
        return value

    def _build_request_kwargs(self, api_cfg: ApiConfig) -> dict[str, Any]:
        """Build httpx request kwargs from ``spec.api``."""
        json_payload = api_cfg.json_body
        body = api_cfg.body
        data_payload = api_cfg.data

        if json_payload is not None and body is not None:
            raise ExecutionError(
                "spec.api.json and spec.api.body are mutually exclusive"
            )

        request_kwargs: dict[str, Any] = {}
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        elif body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = body
        elif data_payload is not None:
            request_kwargs["data"] = data_payload
        return request_kwargs

    def _parse_response(
        self,
        resp: httpx.Response,
        *,
        response_cfg: ApiResponseConfig,
        max_body_bytes: int,
        idx: int,
        prompt_str: str,
    ) -> APIItem:
        """Turn one HTTP response into an APIItem, applying response config."""
        body_bytes = resp.content
        truncated = False
        if max_body_bytes is not None and len(body_bytes) > max_body_bytes:
            body_bytes = body_bytes[:max_body_bytes]
            truncated = True

        item = APIItem(
            index=idx,
            url=str(resp.url),
            status_code=resp.status_code,
            truncated=truncated,
            prompt=prompt_str,
        )

        if response_cfg.include_headers:
            item.headers = dict(resp.headers)

        body_text: str | None = None
        if response_cfg.return_body:
            encoding = resp.encoding or "utf-8"
            body_text = body_bytes.decode(encoding, errors="replace")

        if response_cfg.parse_json:
            item.response_json = resp.json()
            if not isinstance(item.response_json, dict):
                raise ExecutionError("Response is not a valid JSON mapping")
            usage = item.response_json.get("usage")
            if isinstance(usage, dict):
                item.usage = usage
            try:
                item.text = item.response_json["choices"][0]["message"]["content"]
            except Exception:
                item.text = None
        elif response_cfg.return_body:
            item.text = body_text

        return item

    def _log_summary(
        self,
        task_id: str,
        total: int,
        failures: int,
        total_retries: int,
        wall: float,
        latencies: list[float],
        sum_prompt: int,
        sum_completion: int,
        sum_reasoning: int,
        backend_counts: dict[str, int],
    ) -> None:
        """Log one summary line for a finished API task."""
        backends = ",".join(
            f"{name}={count}" for name, count in sorted(backend_counts.items())
        )
        logger.info(
            "api summary task=%s calls=%d failures=%d retries=%d wall=%.3fs "
            "latency_p50=%.3fs latency_p95=%.3fs latency_max=%.3fs "
            "prompt_tokens=%d completion_tokens=%d reasoning_tokens=%d "
            "backends=%s",
            task_id,
            total,
            failures,
            total_retries,
            wall,
            _percentile(latencies, 50),
            _percentile(latencies, 95),
            max(latencies) if latencies else 0.0,
            sum_prompt,
            sum_completion,
            sum_reasoning,
            backends or "-",
        )

    def run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        with self._cancel_lock:
            self._active_task_id = task.task_id
            # Event is set iff this task's id was pending; other ids are stale.
            if task.task_id not in self._pending_cancelled_ids:
                self._cancel_event.clear()
            self._pending_cancelled_ids.clear()
        try:
            return self._run(task, out_dir)
        finally:
            with self._cancel_lock:
                self._active_task_id = None

    def _run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        spec = self.require_spec(task, ApiSpecStrict)
        api_cfg = spec.api or ApiConfig.model_validate({})

        url = api_cfg.url
        method = api_cfg.method
        headers = api_cfg.headers or {}

        if url is None:
            url = os.getenv("NEBULA_API_BASE_URL")
            if not url:
                raise ExecutionError("spec.api.url or NEBULA_API_BASE_URL is required")
            url = url.rstrip("/") + "/v1/chat/completions"

            if not any(is_credential_key(k) for k in headers):
                token = os.getenv("NEBULA_API_TOKEN")
                if not token:
                    raise ExecutionError(
                        "no credential configured: set an Authorization header or "
                        "NEBULA_API_TOKEN"
                    )
                headers["Authorization"] = f"Bearer {token}"

        params = api_cfg.params

        timeout = httpx.Timeout(api_cfg.timeout_sec)

        verify_tls = api_cfg.verify_tls
        follow_redirects = api_cfg.follow_redirects

        response_cfg = api_cfg.response or ApiResponseConfig.model_validate({})
        max_body_bytes = response_cfg.max_body_bytes
        raise_for_status = response_cfg.raise_for_status

        retries = api_cfg.retries
        concurrency = api_cfg.concurrency

        base = self._base_url(str(url))
        client = self._get_client(
            base, timeout, verify_tls, follow_redirects, concurrency
        )

        entry = self._collect_prompts_for_spec(spec, task_id=task.task_id)
        prompts = entry.prompts
        if not prompts and not entry.tables:
            raise ExecutionError("spec.data produced no rows")

        request_kwargs = self._build_request_kwargs(api_cfg)

        failed = threading.Event()
        first_error: list[BaseException] = []

        total = len(prompts)
        done = 0
        failures = 0
        total_retries = 0
        latencies: list[float] = []
        sum_prompt = 0
        sum_completion = 0
        sum_reasoning = 0
        backend_counts: dict[str, int] = {}
        task_start = time.monotonic()
        in_flight: dict[int, float] = {}
        in_flight_lock = threading.Lock()
        stats_lock = threading.Lock()

        def _record_call(
            idx: int,
            attempts: int,
            status: Any,
            start: float,
            body: Any,
            *,
            failed: bool,
        ) -> None:
            nonlocal done, failures, total_retries
            nonlocal sum_prompt, sum_completion, sum_reasoning
            wall = time.monotonic() - start
            with in_flight_lock:
                in_flight.pop(idx, None)
            (
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                finish_reason,
                backend,
            ) = _chat_completion_stats(body)
            with stats_lock:
                done += 1
                if failed:
                    failures += 1
                total_retries += max(0, attempts - 1)
                latencies.append(wall)
                if prompt_tokens is not None:
                    sum_prompt += prompt_tokens
                if completion_tokens is not None:
                    sum_completion += completion_tokens
                if reasoning_tokens is not None:
                    sum_reasoning += reasoning_tokens
                if backend is not None:
                    backend_counts[backend] = backend_counts.get(backend, 0) + 1
            logger.info(
                "api call task=%s row=%d attempts=%d status=%s wall=%.3fs "
                "prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s "
                "finish_reason=%s backend=%s",
                task.task_id,
                idx,
                attempts,
                _fmt(status),
                wall,
                _fmt(prompt_tokens),
                _fmt(completion_tokens),
                _fmt(reasoning_tokens),
                _fmt(finish_reason),
                _fmt(backend),
            )

        def _issue(idx: int, prompt: Any) -> APIItem:
            if self._cancel_event.is_set():
                raise TaskCancelledError("API task cancelled")
            if failed.is_set():
                raise ExecutionError(f"API task failed on an earlier row (row {idx})")
            prompt_str = self._prompt_to_str(prompt)
            kwargs = self._substitute_prompt(request_kwargs, prompt)
            start = time.monotonic()
            with in_flight_lock:
                in_flight[idx] = start
            try:
                resp, attempts = self._request_with_retries(
                    client,
                    method,
                    str(url),
                    headers,
                    params,
                    kwargs,
                    retries,
                    failed,
                )
            except _RequestFailed as exc:
                _record_call(
                    idx,
                    exc.attempts,
                    exc.error.__class__.__name__,
                    start,
                    None,
                    failed=True,
                )
                error = ExecutionError(
                    f"API request failed (row {idx}): {exc.error}", retryable=True
                )
                if not first_error:
                    first_error.append(error)
                failed.set()
                raise error from exc.error
            except TaskCancelledError:
                raise
            except BaseException as exc:
                if not first_error:
                    first_error.append(exc)
                failed.set()
                raise

            if raise_for_status and resp.is_error:
                message = f"API request returned status {resp.status_code} (row {idx})"
                body_text = resp.text[:200]
                if body_text:
                    message = f"{message}: {body_text}"
                retryable = _is_retryable_status(resp.status_code)
                _record_call(idx, attempts, resp.status_code, start, None, failed=True)
                error = ExecutionError(message, retryable=retryable)
                if not first_error:
                    first_error.append(error)
                failed.set()
                raise error

            try:
                item = self._parse_response(
                    resp,
                    response_cfg=response_cfg,
                    max_body_bytes=max_body_bytes,
                    idx=idx,
                    prompt_str=prompt_str,
                )
            except BaseException as exc:
                if not first_error:
                    first_error.append(exc)
                failed.set()
                raise

            if self._cancel_event.is_set():
                raise TaskCancelledError("API task cancelled")

            _record_call(
                idx,
                attempts,
                resp.status_code,
                start,
                item.response_json,
                failed=False,
            )
            return item

        results: dict[int, APIItem] = {}
        heartbeat_stop = threading.Event()

        def _heartbeat() -> None:
            while not heartbeat_stop.wait(60):
                with in_flight_lock:
                    outstanding = dict(in_flight)
                if not outstanding:
                    continue
                with stats_lock:
                    done_snapshot = done
                oldest = min(outstanding.values())
                logger.info(
                    "api heartbeat task=%s done=%d/%d in_flight=%d oldest_age=%.0fs",
                    task.task_id,
                    done_snapshot,
                    total,
                    len(outstanding),
                    time.monotonic() - oldest,
                )

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {}
                for idx, prompt in enumerate(prompts):
                    if self._cancel_event.is_set():
                        raise TaskCancelledError("API task cancelled")
                    if failed.is_set():
                        break
                    futures[pool.submit(_issue, idx, prompt)] = idx
                for future in as_completed(futures):
                    idx = futures[future]
                    if self._cancel_event.is_set():
                        raise TaskCancelledError("API task cancelled")
                    try:
                        results[idx] = future.result()
                    except TaskCancelledError:
                        raise
                    except BaseException:
                        pass
                if first_error:
                    raise first_error[0]
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=5)
            wall = time.monotonic() - task_start
            with stats_lock:
                failures_snapshot = failures
                retries_snapshot = total_retries
                latencies_snapshot = list(latencies)
                prompt_snapshot = sum_prompt
                completion_snapshot = sum_completion
                reasoning_snapshot = sum_reasoning
                backends_snapshot = dict(backend_counts)
            self._log_summary(
                task.task_id,
                total,
                failures_snapshot,
                retries_snapshot,
                wall,
                latencies_snapshot,
                prompt_snapshot,
                completion_snapshot,
                reasoning_snapshot,
                backends_snapshot,
            )

        if self._cancel_event.is_set():
            raise TaskCancelledError("API task cancelled")

        items = [results[idx] for idx in range(len(prompts))]

        result_items: list[APIItem | APIGroupItem] = []
        if entry.tables:
            # One result item per table, sliced as in DataMixin._populate_table.
            grouped: list[APIGroupItem] = []
            cur = 0
            for group_index, df in enumerate(entry.tables):
                size = len(df)
                grouped.append(
                    APIGroupItem(index=group_index, rows=items[cur : cur + size])
                )
                cur += size
            if cur != len(items):
                raise ExecutionError(
                    f"Output length {len(items)} does not match "
                    f"the total number of rows {cur} in table stores."
                )
            result_items.extend(grouped)
        else:
            result_items.extend(items)

        if result_items:
            first = result_items[0]
            if isinstance(first, APIGroupItem):
                status_code = next(
                    (
                        r.status_code
                        for g in result_items
                        if isinstance(g, APIGroupItem)
                        for r in g.rows
                    ),
                    0,
                )
                truncated = any(
                    r.truncated
                    for g in result_items
                    if isinstance(g, APIGroupItem)
                    for r in g.rows
                )
            else:
                status_code = first.status_code
                truncated = any(
                    item.truncated for item in result_items if isinstance(item, APIItem)
                )
        else:
            status_code = 0
            truncated = False

        return APIResult(
            ok=True,
            executor=self.name,
            method=method,
            url=str(url),
            status_code=status_code,
            truncated=truncated,
            items=result_items,
        )
