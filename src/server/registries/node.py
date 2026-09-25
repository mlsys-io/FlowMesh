import asyncio
import logging
import time
from collections.abc import Sequence
from threading import Thread
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)
from redis.client import PubSub

from shared.schemas.command import (
    CommandMessage,
    CommandResponse,
    CommandType,
)
from shared.schemas.node import NodeInfo
from shared.utils import new_node_id

from .. import env
from ..clients.redis import (
    NODE_ALIAS_LEASE_PREFIX,
    NODE_ID_SEQ_KEY,
    NODE_RESPONSE_CHANNEL,
    NODES_SET_KEY,
    REDIS_CONN_ERRORS,
    RedisClient,
    iter_pubsub_messages,
    node_alias_lease_key,
    node_cmd_channel,
    node_hb_key,
    node_key,
)

_RECONNECT_BACKOFF_SEC = 1.0

# A node alias lease is a hash {node_id, ttl_ms} expiring after ttl_ms. It is
# stale once its holder has not refreshed it for half its TTL; the age is read
# from PTTL, so no clocks are compared.
_REGISTER_LUA = """
local holder = redis.call('HGET', KEYS[1], 'node_id')
if holder and holder ~= ARGV[1] then
  local held_ttl = tonumber(redis.call('HGET', KEYS[1], 'ttl_ms')) or 0
  local remaining = redis.call('PTTL', KEYS[1])
  if remaining >= 0 and remaining * 2 >= held_ttl then
    return holder
  end
end
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], 'node_id', ARGV[1], 'ttl_ms', ARGV[2])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('SADD', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[3], unpack(ARGV, 3))
return ''
"""

_HEARTBEAT_LUA = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
  return ''
end
redis.call('SET', KEYS[3], ARGV[2], 'EX', ARGV[3])
if ARGV[5] ~= '' then
  redis.call('HSET', KEYS[2], 'last_seen', ARGV[2], 'current_gpu_count', ARGV[5])
else
  redis.call('HSET', KEYS[2], 'last_seen', ARGV[2])
end
local alias = redis.call('HGET', KEYS[2], 'alias')
if not alias then
  return ''
end
local lease = ARGV[4] .. alias
local holder = redis.call('HGET', lease, 'node_id')
if holder and holder ~= ARGV[1] then
  return holder
end
local ttl_ms = tonumber(ARGV[3]) * 1000
redis.call('HSET', lease, 'node_id', ARGV[1], 'ttl_ms', ttl_ms)
redis.call('PEXPIRE', lease, ttl_ms)
return ''
"""

_UNREGISTER_LUA = """
local alias = redis.call('HGET', KEYS[2], 'alias')
if alias then
  local lease = ARGV[2] .. alias
  if redis.call('HGET', lease, 'node_id') == ARGV[1] then
    redis.call('DEL', lease)
  end
end
redis.call('SREM', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[2], KEYS[3])
return ''
"""


class NodeAliasInUseError(Exception):
    """A live node other than the registrant holds the requested node alias."""

    def __init__(self, alias: str, holder: str) -> None:
        super().__init__(
            f"node alias '{alias}' is held by live node {holder}; "
            "set a distinct NODE_ALIAS"
        )
        self.alias = alias
        self.holder = holder


class Node(BaseModel):
    id: str
    namespace: str
    cluster: str
    alias: str
    version: str | None = None
    started_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    last_seen: str | None = None
    max_gpu_count: int = 0
    current_gpu_count: int = 0

    @classmethod
    def from_info(cls, node_id: str, info: NodeInfo) -> "Node":
        return cls(
            id=node_id,
            namespace=info.namespace,
            cluster=info.cluster,
            alias=info.alias,
            version=info.version,
            started_at=info.started_at,
            tags=info.tags,
            last_seen=info.last_seen,
            max_gpu_count=info.max_gpu_count,
            current_gpu_count=info.max_gpu_count,
        )

    @field_serializer("tags")
    def serialize_tags(self, tags: list[str]) -> str:
        return ",".join(tags)

    @field_validator("tags", mode="before")
    def validate_tags(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_str = v.strip()
            return v_str.split(",") if v_str else []
        return v


class NodeRegistry:
    def __init__(self, rds: RedisClient, logger: logging.Logger) -> None:
        self.logger = logger
        self._rds = rds

        self._pubsub: PubSub | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

        self._node_responses: dict[str, asyncio.Future[CommandResponse]] = {}

    def _node_from_info(self, node_id: str, node_info: NodeInfo) -> Node:
        return Node.from_info(node_id, node_info)

    def start(self) -> Thread:
        if self._pubsub is not None:
            raise RuntimeError("Node registry pubsub already initialized")
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "Node registry must be started inside an event loop: %s", exc
            )
        assert not self._running
        self._running = True
        self._pubsub = self._rds.sync.subscribe_control(NODE_RESPONSE_CHANNEL)
        thread = Thread(
            target=self._run,
            name="NodeRegistryThread",
            daemon=True,
        )
        thread.start()
        return thread

    def shutdown(self) -> None:
        self._running = False
        if pubsub := self._pubsub:
            try:
                pubsub.close()
            except REDIS_CONN_ERRORS:
                pass
        self._loop = None

    def _resubscribe(self) -> bool:
        """Re-establish the response subscription after a dropped connection,
        retrying with backoff until it succeeds or the registry is shut down."""
        pubsub = self._pubsub
        if pubsub is None:
            raise RuntimeError("Node registry pubsub not initialized")
        try:
            pubsub.close()
        except REDIS_CONN_ERRORS:
            pass
        self.logger.warning("Node registry pubsub dropped; reconnecting")
        while self._running:
            time.sleep(_RECONNECT_BACKOFF_SEC)
            try:
                self._pubsub = self._rds.sync.subscribe_control(NODE_RESPONSE_CHANNEL)
            except REDIS_CONN_ERRORS as exc:
                self.logger.warning(
                    "Node registry resubscribe failed (%s); retrying", exc
                )
                continue
            self.logger.info("Node registry reconnected")
            return True
        return False

    def _run(self) -> None:
        if self._loop is None:
            self.logger.error("Node registry not started")
            return
        try:
            while self._running and (pubsub := self._pubsub):
                try:
                    for data in iter_pubsub_messages(pubsub):
                        try:
                            cmd = CommandResponse.model_validate(data)
                        except ValidationError as exc:
                            self.logger.error("Invalid node response: %s", exc)
                            continue
                        self.set_node_response(cmd)
                except Exception as exc:
                    if self._running:
                        self.logger.exception("Node registry error: %s", exc)
                if not self._running:
                    break
                self._resubscribe()
        finally:
            pubsub = self._pubsub
            self._pubsub = None
            if pubsub is not None:
                try:
                    pubsub.close()
                except REDIS_CONN_ERRORS:
                    pass

    # ------------------------------------------------------------------ #
    # Node lifecycle helpers
    # ------------------------------------------------------------------ #

    def register_node(self, node_info: NodeInfo) -> str:
        """Register a node under a fresh id, taking its alias lease.

        Raises `NodeAliasInUseError` when another live node holds the alias.
        """
        node_id = self._allocate_node_id()
        keys, args = self._register_script_args(node_id, node_info)
        holder = self._rds.sync.eval(_REGISTER_LUA, len(keys), *keys, *args)
        if holder:
            raise NodeAliasInUseError(node_info.alias, holder)
        return node_id

    async def register_node_async(self, node_info: NodeInfo) -> str:
        node_id = await self._allocate_node_id_async()
        keys, args = self._register_script_args(node_id, node_info)
        holder = await self._rds.asyncio.eval(_REGISTER_LUA, len(keys), *keys, *args)
        if holder:
            raise NodeAliasInUseError(node_info.alias, holder)
        return node_id

    def update_node_hb(
        self,
        node_id: str,
        ts: str,
        ttl_sec: int,
        current_gpu_count: int | None = None,
    ) -> str | None:
        """Refresh a registered node's heartbeat and alias lease.

        Returns the id of another node holding this node's alias, if any.
        """
        keys, args = _heartbeat_script_args(node_id, ts, ttl_sec, current_gpu_count)
        return self._rds.sync.eval(_HEARTBEAT_LUA, len(keys), *keys, *args) or None

    async def update_node_hb_async(
        self,
        node_id: str,
        ts: str,
        ttl_sec: int,
        current_gpu_count: int | None = None,
    ) -> str | None:
        keys, args = _heartbeat_script_args(node_id, ts, ttl_sec, current_gpu_count)
        holder = await self._rds.asyncio.eval(_HEARTBEAT_LUA, len(keys), *keys, *args)
        return holder or None

    def unregister_node(self, node_id: str) -> None:
        keys, args = _unregister_script_args(node_id)
        self._rds.sync.eval(_UNREGISTER_LUA, len(keys), *keys, *args)

    async def unregister_node_async(self, node_id: str) -> None:
        keys, args = _unregister_script_args(node_id)
        await self._rds.asyncio.eval(_UNREGISTER_LUA, len(keys), *keys, *args)

    # ------------------------------------------------------------------ #
    # Node query helpers
    # ------------------------------------------------------------------ #

    def node_exists(self, node_id: str) -> bool:
        return self._rds.sync.exists(node_key(node_id))

    async def node_exists_async(self, node_id: str) -> bool:
        return await self._rds.asyncio.exists(node_key(node_id))

    def get_node_ids(self) -> set[str]:
        return self._rds.sync.set_members(NODES_SET_KEY)

    async def get_node_ids_async(self) -> set[str]:
        return await self._rds.asyncio.set_members(NODES_SET_KEY)

    def get_node(self, node_id: str) -> Node | None:
        data = self._rds.sync.hash_getall(node_key(node_id))
        return Node.model_validate(data) if data else None

    async def get_node_async(self, node_id: str) -> Node | None:
        data = await self._rds.asyncio.hash_getall(node_key(node_id))
        return Node.model_validate(data) if data else None

    def list_nodes(self) -> list[Node]:
        node_ids = self.get_node_ids()
        nodes: list[Node] = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes.append(node)
        return nodes

    async def list_nodes_async(self) -> list[Node]:
        node_ids = await self.get_node_ids_async()
        nodes: list[Node] = []
        for node_id in node_ids:
            node = await self.get_node_async(node_id)
            if node:
                nodes.append(node)
        return nodes

    def get_nodes(self, node_ids: Sequence[str]) -> list[Node | None]:
        with self._rds.sync.control_pipeline() as pipe:
            for node_id in node_ids:
                pipe.hgetall(node_key(node_id))
            raws: list[dict[str, Any]] = pipe.execute()
        results: list[Node | None] = []
        for raw in raws:
            if not raw:
                results.append(None)
                continue
            try:
                results.append(Node.model_validate(raw))
            except ValidationError:
                results.append(None)
        return results

    async def get_nodes_async(self, node_ids: Sequence[str]) -> list[Node | None]:
        async with self._rds.asyncio.control_pipeline() as pipe:
            for node_id in node_ids:
                pipe.hgetall(node_key(node_id))
            raws: list[dict[str, Any]] = await pipe.execute()
        results: list[Node | None] = []
        for raw in raws:
            if not raw:
                results.append(None)
                continue
            try:
                results.append(Node.model_validate(raw))
            except ValidationError:
                results.append(None)
        return results

    # ------------------------------------------------------------------ #
    # Node command helpers
    # ------------------------------------------------------------------ #

    def send_node_cmd(
        self, node_id: str, cmd: CommandMessage
    ) -> asyncio.Future[CommandResponse]:
        if self._loop is None:
            raise RuntimeError("Node registry not started")

        channel = node_cmd_channel(node_id)
        fut = self._loop.create_future()
        self._node_responses[cmd.command_id] = fut
        try:
            self._rds.sync.publish_control(channel, cmd.model_dump_json())
        except Exception:
            self._node_responses.pop(cmd.command_id, None)
            fut.cancel()
            raise
        return fut

    async def exec_node_cmd(
        self, node_id: str, cmd: CommandMessage, timeout: float | None = None
    ) -> CommandResponse:
        if not await self.node_exists_async(node_id):
            raise ValueError("Node not found")

        fut = self.send_node_cmd(node_id, cmd)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as exc:
            self._node_responses.pop(cmd.command_id, None)
            fut.cancel()
            raise TimeoutError("Node command timed out") from exc

    def is_node_alive(self, node_id: str) -> bool:
        """Return True if the node heartbeat key has not expired."""
        return self._rds.sync.exists(node_hb_key(node_id))

    async def create_worker_on_node(
        self,
        node_id: str,
        gpu_count: int = 0,
        worker_alias: str | None = None,
        timeout: float = 120.0,
    ) -> CommandResponse:
        """Send CREATE_WORKER_ON_NODE command to a node and wait for response."""
        cmd = CommandMessage(
            command=CommandType.CREATE_WORKER_ON_NODE,
            payload={"gpu_count": gpu_count, "worker_alias": worker_alias or ""},
        )
        return await self.exec_node_cmd(node_id, cmd, timeout=timeout)

    async def destroy_worker_on_node(
        self,
        node_id: str,
        worker_name: str,
        timeout: float = 60.0,
    ) -> CommandResponse:
        """Send DESTROY_WORKER command to a node and wait for its response."""
        cmd = CommandMessage(
            command=CommandType.DESTROY_WORKER,
            payload={"worker_name": worker_name},
        )
        return await self.exec_node_cmd(node_id, cmd, timeout=timeout)

    def set_node_response(self, resp: CommandResponse) -> None:
        if self._loop is None:
            raise RuntimeError("Node registry not started")
        fut = self._node_responses.pop(resp.command_id, None)
        if fut is None:
            self.logger.warning(
                "Received response for unknown command id %s", resp.command_id
            )
            return
        self._loop.call_soon_threadsafe(fut.set_result, resp)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _register_script_args(
        self, node_id: str, node_info: NodeInfo
    ) -> tuple[list[str], list[str]]:
        node = self._node_from_info(node_id, node_info)
        fields = [
            str(item)
            for key, value in node.model_dump().items()
            if value is not None
            for item in (key, value)
        ]
        keys = [node_alias_lease_key(node_info.alias), NODES_SET_KEY, node_key(node_id)]
        lease_ttl_ms = str(env.SERVER_HEARTBEAT_TTL * 1000)
        return keys, [node_id, lease_ttl_ms, *fields]

    def _allocate_node_id(self) -> str:
        seq = self._rds.sync.incr(NODE_ID_SEQ_KEY)
        return new_node_id(seq)

    async def _allocate_node_id_async(self) -> str:
        seq = await self._rds.asyncio.incr(NODE_ID_SEQ_KEY)
        return new_node_id(seq)


def _heartbeat_script_args(
    node_id: str, ts: str, ttl_sec: int, current_gpu_count: int | None
) -> tuple[list[str], list[str]]:
    keys = [NODES_SET_KEY, node_key(node_id), node_hb_key(node_id)]
    gpu_count = "" if current_gpu_count is None else str(current_gpu_count)
    return keys, [node_id, ts, str(ttl_sec), NODE_ALIAS_LEASE_PREFIX, gpu_count]


def _unregister_script_args(node_id: str) -> tuple[list[str], list[str]]:
    keys = [NODES_SET_KEY, node_key(node_id), node_hb_key(node_id)]
    return keys, [node_id, NODE_ALIAS_LEASE_PREFIX]
