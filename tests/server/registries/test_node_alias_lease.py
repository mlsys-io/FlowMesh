"""Node alias leases keep `node_alias` unique among live nodes."""

import logging
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    NODES_SET_KEY,
    AsyncRedisClient,
    RedisClient,
    SyncRedisClient,
    node_alias_lease_key,
    node_hb_key,
    node_key,
)
from server.registries.node import NodeAliasInUseError, NodeRegistry
from shared.schemas.node import NodeInfo

TTL_SEC = 120
LEASE = node_alias_lease_key("gpu-a")


def _info(alias: str = "gpu-a") -> NodeInfo:
    return NodeInfo(
        namespace="ns",
        cluster="cl",
        alias=alias,
        version="0.1.0",
        started_at="2026-09-25T00:00:00Z",
        tags=[],
        last_seen="2026-09-25T00:00:00Z",
        max_gpu_count=0,
    )


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def rds(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def registry(
    server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
) -> NodeRegistry:
    monkeypatch.setattr("server.env.SERVER_HEARTBEAT_TTL", TTL_SEC)
    sync = SyncRedisClient.__new__(SyncRedisClient)
    sync._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    async_client = AsyncRedisClient.__new__(AsyncRedisClient)
    cast(Any, async_client)._control = fakeredis.FakeAsyncRedis(
        server=server, decode_responses=True
    )
    client = RedisClient.__new__(RedisClient)
    client.sync = sync
    client.asyncio = async_client
    return NodeRegistry(client, logging.getLogger("test.node_alias_lease"))


def _pttl(rds: fakeredis.FakeRedis) -> int:
    return cast(int, rds.pttl(LEASE))


def _age_lease(rds: fakeredis.FakeRedis, elapsed_sec: float) -> None:
    ttl_ms = int(cast(str, rds.hget(LEASE, "ttl_ms")))
    rds.pexpire(LEASE, ttl_ms - int(elapsed_sec * 1000))


def test_register_takes_lease_and_writes_record(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    node_id = registry.register_node(_info())

    assert rds.hget(LEASE, "node_id") == node_id
    assert 0 < _pttl(rds) <= TTL_SEC * 1000
    assert rds.sismember(NODES_SET_KEY, node_id)
    assert rds.hget(node_key(node_id), "alias") == "gpu-a"


def test_duplicate_alias_is_refused_without_writing(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    holder = registry.register_node(_info())

    with pytest.raises(NodeAliasInUseError) as exc:
        registry.register_node(_info())

    assert exc.value.holder == holder
    assert "NODE_ALIAS" in str(exc.value)
    assert rds.smembers(NODES_SET_KEY) == {holder}


@pytest.mark.asyncio
async def test_async_register_refuses_duplicate(registry: NodeRegistry) -> None:
    await registry.register_node_async(_info())
    with pytest.raises(NodeAliasInUseError):
        await registry.register_node_async(_info())


def test_distinct_aliases_coexist(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    id_a = registry.register_node(_info("gpu-a"))
    id_b = registry.register_node(_info("gpu-b"))

    assert rds.hget(node_alias_lease_key("gpu-a"), "node_id") == id_a
    assert rds.hget(node_alias_lease_key("gpu-b"), "node_id") == id_b


def test_stale_holder_is_taken_over(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    registry.register_node(_info())
    _age_lease(rds, TTL_SEC / 2 + 1)

    node_id = registry.register_node(_info())

    assert rds.hget(LEASE, "node_id") == node_id


def test_recently_refreshed_holder_is_not_taken_over(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    registry.register_node(_info())
    _age_lease(rds, TTL_SEC / 2 - 5)

    with pytest.raises(NodeAliasInUseError):
        registry.register_node(_info())


def test_expired_lease_is_free(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    registry.register_node(_info())
    rds.delete(LEASE)

    node_id = registry.register_node(_info())

    assert rds.hget(LEASE, "node_id") == node_id


def test_heartbeat_refreshes_own_lease_with_node_ttl(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    node_id = registry.register_node(_info())
    _age_lease(rds, 50)

    holder = registry.update_node_hb(node_id, "ts", 300, current_gpu_count=2)

    assert holder is None
    assert rds.hget(LEASE, "ttl_ms") == "300000"
    assert _pttl(rds) > 290_000
    assert rds.get(node_hb_key(node_id)) == "ts"
    assert rds.hget(node_key(node_id), "current_gpu_count") == "2"


def test_heartbeat_retakes_lapsed_lease(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    node_id = registry.register_node(_info())
    rds.delete(LEASE)

    assert registry.update_node_hb(node_id, "ts", TTL_SEC) is None
    assert rds.hget(LEASE, "node_id") == node_id


@pytest.mark.asyncio
async def test_heartbeat_reports_lease_taken_over(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    old_id = registry.register_node(_info())
    _age_lease(rds, TTL_SEC)
    new_id = registry.register_node(_info())

    assert await registry.update_node_hb_async(old_id, "ts", TTL_SEC) == new_id
    assert rds.hget(LEASE, "node_id") == new_id


def test_heartbeat_of_unregistered_node_writes_nothing(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    node_id = registry.register_node(_info())
    registry.unregister_node(node_id)

    assert registry.update_node_hb(node_id, "ts", TTL_SEC) is None
    assert not rds.exists(node_key(node_id))
    assert not rds.exists(node_hb_key(node_id))
    assert not rds.exists(LEASE)


def test_unregister_releases_lease_for_immediate_reuse(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    node_id = registry.register_node(_info())
    registry.update_node_hb(node_id, "ts", TTL_SEC)

    registry.unregister_node(node_id)

    assert not rds.exists(LEASE)
    assert not rds.sismember(NODES_SET_KEY, node_id)
    assert not rds.exists(node_key(node_id))
    assert not rds.exists(node_hb_key(node_id))
    registry.register_node(_info())


@pytest.mark.asyncio
async def test_unregister_never_releases_another_nodes_lease(
    registry: NodeRegistry, rds: fakeredis.FakeRedis
) -> None:
    old_id = registry.register_node(_info())
    _age_lease(rds, TTL_SEC)
    new_id = registry.register_node(_info())

    await registry.unregister_node_async(old_id)

    assert rds.hget(LEASE, "node_id") == new_id
