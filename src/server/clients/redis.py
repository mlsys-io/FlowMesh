import json
import logging
import socket
import ssl
from collections.abc import Awaitable, Iterable
from typing import Any
from urllib.parse import urlparse, urlunparse

import redis
import redis.asyncio as async_redis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection
from redis.client import Pipeline, PubSub
from redis.connection import SSLConnection as SyncSSLConnection
from redis.typing import EncodableT


def _keepalive_kwargs() -> dict[str, Any]:
    """Connection kwargs that keep an idle Redis socket alive.

    An idle pub/sub SUBSCRIBE gets culled by intermediaries (load balancers, NAT
    gateways), dropping a node's dispatch subscription. Keepalive probes hold it open
    and ``health_check_interval`` reconnects a dead socket instead of blocking on it.
    """
    options: dict[int, int] = {}
    # These constants are Linux-only; skip any the running platform lacks (e.g.
    # macOS/Windows dev hosts) so importing this module there doesn't fail.
    for name, value in (
        ("TCP_KEEPIDLE", 60),
        ("TCP_KEEPINTVL", 15),
        ("TCP_KEEPCNT", 4),
    ):
        opt = getattr(socket, name, None)
        if opt is not None:
            options[opt] = value
    return {
        "socket_keepalive": True,
        "socket_keepalive_options": options,
        "health_check_interval": 30,
    }


TASK_EVENT_STREAM_KEY = "tasks:events:stream"
TASK_EVENT_CURSOR_KEY = "tasks:events:cursor"
TASK_EVENT_STREAM_MAXLEN = 100_000

WORKFLOWS_SET_KEY = "workflows:ids"

TASK_LOGS_STREAM_PREFIX = "logs:task:"
WORKFLOW_LOGS_STREAM_PREFIX = "logs:workflow:"

WORKERS_SET_KEY = "workers:ids"
WORKER_ID_SEQ_KEY = "workers:id_seq"
WORKER_EVENT_CHANNEL = "workers:events"

NODES_SET_KEY = "nodes:ids"
NODE_ID_SEQ_KEY = "nodes:id_seq"
NODE_EVENT_CHANNEL = "nodes:events"
NODE_RESPONSE_CHANNEL = "nodes:responses"


def workflow_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}"


def workflow_tasks_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:tasks"


def workflow_dispatched_tasks_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:dispatched_tasks"


def workflow_failed_tasks_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:failed_tasks"


def workflow_cancelled_tasks_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:cancelled_tasks"


def workflow_sched_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:sched"


def task_state_key(task_id: str) -> str:
    return f"task:{task_id}:state"


def worker_key(worker_id: str) -> str:
    return f"worker:{worker_id}"


def worker_hb_key(worker_id: str) -> str:
    return f"worker:{worker_id}:hb"


def node_key(node_id: str) -> str:
    return f"node:{node_id}"


def node_hb_key(node_id: str) -> str:
    return f"node:{node_id}:hb"


def node_dispatch_channel(node_id: str) -> str:
    return f"node:{node_id}:dispatch"


def node_cmd_channel(node_id: str) -> str:
    return f"node:{node_id}:cmds"


def task_log_stream_key(task_id: str) -> str:
    return TASK_LOGS_STREAM_PREFIX + task_id


def workflow_log_stream_key(workflow_id: str) -> str:
    return WORKFLOW_LOGS_STREAM_PREFIX + workflow_id


def task_log_archive_last_id_key(task_id: str) -> str:
    return f"task:{task_id}:logs:archived_last_id"


def task_log_closed_key(task_id: str) -> str:
    return f"task:{task_id}:logs:closed"


def workflow_log_closed_key(workflow_id: str) -> str:
    return f"workflow:{workflow_id}:logs:closed"


def ssh_up_key(relay_token: str) -> str:
    return f"ssh:relay:{relay_token}:up"


def ssh_down_key(relay_token: str) -> str:
    return f"ssh:relay:{relay_token}:down"


def ssh_connection_key(connection_id: str) -> str:
    return f"ssh:connection:{connection_id}"


SSH_CONNECTION_IDS_KEY = "ssh:connections:active"


def parse_pubsub_message(msg: dict[str, Any] | None) -> Any | None:
    """Decode a redis-py pub/sub frame into its JSON payload.

    Returns ``None`` for control frames, empty payloads, and malformed JSON. Payloads
    are always JSON objects, so a ``None`` return means "nothing to deliver".
    """
    if msg is None or msg.get("type") != "message":
        return None
    raw = msg.get("data")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def iter_pubsub_messages(pubsub: PubSub) -> Iterable[Any]:
    """Iterate over messages from a Redis PubSub instance.

    Stops cleanly on pubsub teardown (`ConnectionError`, `OSError`); skips
    individual malformed JSON payloads without ending iteration so a single
    bad message can't kill the listener.
    """
    try:
        for msg in pubsub.listen():
            parsed = parse_pubsub_message(msg)
            if parsed is None:
                continue
            yield parsed
    except (ConnectionError, OSError):
        return


def _sync[T](value: Awaitable[T] | T) -> T:
    return value  # type: ignore


def _with_redis_auth(url: str, acl_enabled: bool, username: str, password: str) -> str:
    if not acl_enabled:
        return url
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        return url
    if not password:
        raise SystemExit("REDIS_PASSWORD is required when REDIS_ACL_ENABLED=1")
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = f"{username}:{password}@{host}"
    return urlunparse(parsed._replace(netloc=netloc))


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    user = parsed.username or "user"
    netloc = f"{user}:****@{host}"
    return urlunparse(parsed._replace(netloc=netloc))


class SyncRedisClient:
    def __init__(
        self,
        control_url: str,
        telemetry_url: str,
        logger: logging.Logger,
        control_tls_ca_file: str | None = None,
        telemetry_tls_ca_file: str | None = None,
    ) -> None:
        self.control_url = control_url
        self.telemetry_url = telemetry_url
        self.logger = logger
        self._control = self._connect(control_url, "control", control_tls_ca_file)
        self._telemetry = self._connect(
            telemetry_url, "telemetry", telemetry_tls_ca_file
        )

    @property
    def control_client(self) -> redis.Redis:
        """Return the raw control Redis client."""
        return self._control

    @property
    def telemetry_client(self) -> redis.Redis:
        """Return the raw telemetry Redis client."""
        return self._telemetry

    def _redis_ssl_kwargs(self, tls_ca_file: str | None) -> dict[str, Any]:
        if not tls_ca_file:
            return {}
        kwargs: dict[str, Any] = {
            "connection_class": SyncSSLConnection,
            "ssl_cert_reqs": ssl.CERT_REQUIRED,
            "ssl_ca_certs": tls_ca_file,
        }
        return kwargs

    def _connect(self, url: str, label: str, tls_ca_file: str | None) -> redis.Redis:
        ssl_kwargs = self._redis_ssl_kwargs(tls_ca_file)
        try:
            client = redis.from_url(
                url, decode_responses=True, **_keepalive_kwargs(), **ssl_kwargs
            )
            client.ping()
        except Exception as exc:
            self.logger.exception(
                "Failed to connect to %s Redis (%s): %s", label, url, exc
            )
            raise SystemExit(1) from exc
        self.logger.info("Connected to %s Redis: %s", label, _redact_url(url))
        return client

    # ---- String helpers ----
    def get(self, key: str) -> str | None:
        return _sync(self._control.get(key))

    def get_telemetry(self, key: str) -> str | None:
        return _sync(self._telemetry.get(key))

    def mget(self, keys: list[str]) -> list[str | None]:
        return list(_sync(self._control.mget(keys)))

    def mget_telemetry(self, keys: list[str]) -> list[str | None]:
        return list(_sync(self._telemetry.mget(keys)))

    def exists(self, key: str) -> bool:
        return bool(self._control.exists(key))

    def exists_telemetry(self, key: str) -> bool:
        return bool(self._telemetry.exists(key))

    def ttl(self, key: str) -> float | None:
        return _sync(self._control.ttl(key))

    def incr(self, key: str) -> int:
        return int(_sync(self._control.incr(key)))

    def eval(self, script: str, numkeys: int, *keys_and_args: str) -> str:
        return _sync(self._control.eval(script, numkeys, *keys_and_args))

    def set_value(self, key: str, value: str) -> None:
        self._control.set(key, value)

    def set_value_telemetry(self, key: str, value: str) -> None:
        self._telemetry.set(key, value)

    def delete(self, key: str) -> None:
        self._control.delete(key)

    def delete_telemetry(self, key: str) -> None:
        self._telemetry.delete(key)

    def expire(self, key: str, ttl_sec: int) -> bool:
        return bool(self._control.expire(key, max(0, int(ttl_sec))))

    def expire_telemetry(self, key: str, ttl_sec: int) -> bool:
        return bool(self._telemetry.expire(key, max(0, int(ttl_sec))))

    # ---- Hash helpers ----
    def hash_getall(self, key: str) -> dict[str, Any]:
        return _sync(self._control.hgetall(key))

    def hash_mget(self, key: str, fields: list[str]) -> list[Any]:
        return _sync(self._control.hmget(key, fields))

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self._control.hset(key, mapping=mapping)

    # ---- Set helpers ----
    def set_members(self, key: str) -> set[str]:
        return _sync(self._control.smembers(key))

    def set_members_telemetry(self, key: str) -> set[str]:
        return _sync(self._telemetry.smembers(key))

    def sismember(self, key: str, member: str) -> bool:
        return bool(self._control.sismember(key, member))

    def sadd(self, key: str, *members: str) -> None:
        if members:
            self._control.sadd(key, *members)

    def sadd_telemetry(self, key: str, *members: str) -> None:
        if members:
            self._telemetry.sadd(key, *members)

    def srem(self, key: str, *members: str) -> None:
        if members:
            self._control.srem(key, *members)

    def srem_telemetry(self, key: str, *members: str) -> None:
        if members:
            self._telemetry.srem(key, *members)

    # ---- Pipelines ----
    def control_pipeline(self) -> Pipeline:
        return self._control.pipeline()

    # ---- Pub/Sub ----
    def publish_control(self, channel: str, message: str) -> int:
        result = self._control.publish(channel, message)
        return int(_sync(result))

    def publish_telemetry(self, channel: str, message: str) -> int:
        result = self._telemetry.publish(channel, message)
        return int(_sync(result))

    def subscribe_control(self, channel: str) -> PubSub:
        pubsub = self._control.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
        self.logger.info("Subscribed to %s", channel)
        return pubsub

    def subscribe_telemetry(self, channel: str) -> PubSub:
        pubsub = self._telemetry.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
        self.logger.info("Subscribed to %s", channel)
        return pubsub

    # ---- Streams (telemetry) ----
    def xadd_telemetry(
        self,
        key: str,
        fields: dict[EncodableT, EncodableT],
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        return str(
            _sync(
                self._telemetry.xadd(
                    key,
                    fields,
                    maxlen=maxlen,
                    approximate=approximate,
                )
            )
        )

    def xrange_telemetry(
        self, key: str, min_id: str = "-", max_id: str = "+", count: int | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        result = self._telemetry.xrange(key, min=min_id, max=max_id, count=count)
        return list(_sync(result))

    def xread_telemetry(
        self,
        streams: dict[bytes | str | memoryview, int | bytes | str | memoryview],
        count: int | None = None,
        block_ms: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
        result = self._telemetry.xread(streams=streams, count=count, block=block_ms)
        return list(_sync(result))

    # ---- Maintenance ----
    def flush_all(self) -> None:
        self._control.flushdb()
        self.logger.info("Cleared Redis database at %s (control)", self.control_url)
        self._telemetry.flushdb()
        self.logger.info("Cleared Redis database at %s (telemetry)", self.telemetry_url)


def _awaitable[T](value: Awaitable[T] | T) -> Awaitable[T]:
    return value  # type: ignore


class AsyncRedisClient:
    def __init__(
        self,
        control_url: str,
        telemetry_url: str,
        logger: logging.Logger,
        control_tls_ca_file: str | None = None,
        telemetry_tls_ca_file: str | None = None,
    ) -> None:
        self.control_url = control_url
        self.telemetry_url = telemetry_url
        self.logger = logger
        self._control = self._connect(control_url, "control", control_tls_ca_file)
        self._telemetry = self._connect(
            telemetry_url, "telemetry", telemetry_tls_ca_file
        )

    @property
    def control_client(self) -> async_redis.Redis:
        """Return the raw control Redis client."""
        return self._control

    @property
    def telemetry_client(self) -> async_redis.Redis:
        """Return the raw telemetry Redis client."""
        return self._telemetry

    def _redis_ssl_kwargs(self, tls_ca_file: str | None) -> dict[str, Any]:
        if not tls_ca_file:
            return {}
        kwargs: dict[str, Any] = {
            "connection_class": AsyncSSLConnection,
            "ssl_cert_reqs": ssl.CERT_REQUIRED,
            "ssl_ca_certs": tls_ca_file,
        }
        return kwargs

    def _connect(
        self, url: str, label: str, tls_ca_file: str | None
    ) -> async_redis.Redis:
        ssl_kwargs = self._redis_ssl_kwargs(tls_ca_file)
        try:
            client = async_redis.from_url(
                url, decode_responses=True, **_keepalive_kwargs(), **ssl_kwargs
            )
        except Exception as exc:
            self.logger.exception(
                "Failed to connect to %s Redis (%s): %s", label, url, exc
            )
            raise SystemExit(1) from exc
        self.logger.info("Connected to %s Redis: %s", label, _redact_url(url))
        return client

    # ---- String helpers ----
    async def get(self, key: str) -> str | None:
        return await _awaitable(self._control.get(key))

    async def get_telemetry(self, key: str) -> str | None:
        return await _awaitable(self._telemetry.get(key))

    async def mget(self, keys: list[str]) -> list[str | None]:
        return list(await _awaitable(self._control.mget(keys)))

    async def mget_telemetry(self, keys: list[str]) -> list[str | None]:
        return list(await _awaitable(self._telemetry.mget(keys)))

    async def exists(self, key: str) -> bool:
        return bool(await _awaitable(self._control.exists(key)))

    async def exists_telemetry(self, key: str) -> bool:
        return bool(await _awaitable(self._telemetry.exists(key)))

    async def ttl(self, key: str) -> float | None:
        return await _awaitable(self._control.ttl(key))

    async def incr(self, key: str) -> int:
        return int(await _awaitable(self._control.incr(key)))

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> str:
        return await _awaitable(self._control.eval(script, numkeys, *keys_and_args))

    async def set_value(self, key: str, value: str) -> None:
        await _awaitable(self._control.set(key, value))

    async def set_value_telemetry(self, key: str, value: str) -> None:
        await _awaitable(self._telemetry.set(key, value))

    async def delete(self, key: str) -> None:
        await _awaitable(self._control.delete(key))

    async def delete_telemetry(self, key: str) -> None:
        await _awaitable(self._telemetry.delete(key))

    async def expire(self, key: str, ttl_sec: int) -> bool:
        return bool(await _awaitable(self._control.expire(key, max(0, int(ttl_sec)))))

    async def expire_telemetry(self, key: str, ttl_sec: int) -> bool:
        return bool(await _awaitable(self._telemetry.expire(key, max(0, int(ttl_sec)))))

    # ---- Hash helpers ----
    async def hash_getall(self, key: str) -> dict[str, Any]:
        return await _awaitable(self._control.hgetall(key))

    async def hash_mget(self, key: str, fields: list[str]) -> list[Any]:
        return await _awaitable(self._control.hmget(key, fields))

    async def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        await _awaitable(self._control.hset(key, mapping=mapping))

    # ---- Set helpers ----
    async def set_members(self, key: str) -> set[str]:
        return await _awaitable(self._control.smembers(key))

    async def set_members_telemetry(self, key: str) -> set[str]:
        return await _awaitable(self._telemetry.smembers(key))

    async def sismember(self, key: str, member: str) -> bool:
        return bool(await _awaitable(self._control.sismember(key, member)))

    async def sadd(self, key: str, *members: str) -> None:
        if members:
            await _awaitable(self._control.sadd(key, *members))

    async def sadd_telemetry(self, key: str, *members: str) -> None:
        if members:
            await _awaitable(self._telemetry.sadd(key, *members))

    async def srem(self, key: str, *members: str) -> None:
        if members:
            await _awaitable(self._control.srem(key, *members))

    async def srem_telemetry(self, key: str, *members: str) -> None:
        if members:
            await _awaitable(self._telemetry.srem(key, *members))

    # ---- Pipelines ----
    def control_pipeline(self):
        return self._control.pipeline()

    # ---- Pub/Sub ----
    async def publish_control(self, channel: str, message: str) -> int:
        return int(await _awaitable(self._control.publish(channel, message)))

    async def publish_telemetry(self, channel: str, message: str) -> int:
        return int(await _awaitable(self._telemetry.publish(channel, message)))

    async def subscribe_control(self, channel: str) -> AsyncPubSub:
        pubsub = self._control.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel)
        self.logger.info("Subscribed to %s", channel)
        return pubsub

    async def subscribe_telemetry(self, channel: str) -> AsyncPubSub:
        pubsub = self._telemetry.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel)
        self.logger.info("Subscribed to %s", channel)
        return pubsub

    # ---- Streams (telemetry) ----
    async def xadd_telemetry(
        self,
        key: str,
        fields: dict[EncodableT, EncodableT],
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        return str(
            await _awaitable(
                self._telemetry.xadd(
                    key,
                    fields,
                    maxlen=maxlen,
                    approximate=approximate,
                )
            )
        )

    async def xrange_telemetry(
        self,
        key: str,
        min_id: str = "-",
        max_id: str = "+",
        count: int | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        return list(
            await _awaitable(
                self._telemetry.xrange(key, min=min_id, max=max_id, count=count)
            )
        )

    async def xrevrange_telemetry(
        self,
        key: str,
        max_id: str = "+",
        min_id: str = "-",
        count: int | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        return list(
            await _awaitable(
                self._telemetry.xrevrange(key, max=max_id, min=min_id, count=count)
            )
        )

    async def xread_telemetry(
        self,
        streams: dict[bytes | str | memoryview, int | bytes | str | memoryview],
        count: int | None = None,
        block_ms: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
        return list(
            await _awaitable(
                self._telemetry.xread(streams=streams, count=count, block=block_ms)
            )
        )

    async def flush_all(self) -> None:
        await self._control.flushdb()
        self.logger.info("Cleared Redis database at %s (control)", self.control_url)
        await self._telemetry.flushdb()
        self.logger.info("Cleared Redis database at %s (telemetry)", self.telemetry_url)


class RedisClient:
    def __init__(
        self,
        control_url: str,
        telemetry_url: str,
        logger: logging.Logger,
        acl_enabled: bool = False,
        username: str = "admin",
        password: str = "",
        tls_ca_file: str | None = None,
    ) -> None:
        control = _with_redis_auth(
            control_url,
            acl_enabled=acl_enabled,
            username=username,
            password=password,
        )
        telemetry = _with_redis_auth(
            telemetry_url,
            acl_enabled=acl_enabled,
            username=username,
            password=password,
        )
        self.asyncio = AsyncRedisClient(
            control,
            telemetry,
            logger,
            control_tls_ca_file=tls_ca_file,
            telemetry_tls_ca_file=tls_ca_file,
        )
        self.sync = SyncRedisClient(
            control,
            telemetry,
            logger,
            control_tls_ca_file=tls_ca_file,
            telemetry_tls_ca_file=tls_ca_file,
        )
