"""Registry of SSH connections the server is currently relaying."""

import json

from ..clients.redis import SSH_CONNECTION_IDS_KEY, RedisClient, ssh_connection_key
from ..schemas.ssh import SSHConnectionInfo


class SshConnectionRegistry:
    """Tracks the SSH connections a server is relaying."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    async def register_connection(self, info: SSHConnectionInfo) -> None:
        key = ssh_connection_key(info.connection_id)
        payload = info.model_dump_json()
        await self._redis.asyncio.set_value_telemetry(key, payload)
        await self._redis.asyncio.sadd_telemetry(
            SSH_CONNECTION_IDS_KEY, info.connection_id
        )

    async def unregister_connection(self, connection_id: str) -> None:
        key = ssh_connection_key(connection_id)
        await self._redis.asyncio.srem_telemetry(SSH_CONNECTION_IDS_KEY, connection_id)
        await self._redis.asyncio.delete_telemetry(key)

    async def list_connections(self) -> list[SSHConnectionInfo]:
        connection_ids = sorted(
            await self._redis.asyncio.set_members_telemetry(SSH_CONNECTION_IDS_KEY)
        )
        if not connection_ids:
            return []
        raw_values = await self._redis.asyncio.mget_telemetry(
            [ssh_connection_key(connection_id) for connection_id in connection_ids]
        )
        connections: list[SSHConnectionInfo] = []
        stale_ids: list[str] = []
        for connection_id, raw in zip(connection_ids, raw_values, strict=False):
            if not raw:
                stale_ids.append(connection_id)
                continue
            try:
                connections.append(SSHConnectionInfo.model_validate_json(raw))
            except Exception:
                try:
                    connections.append(
                        SSHConnectionInfo.model_validate(json.loads(raw))
                    )
                except Exception:
                    stale_ids.append(connection_id)
        if stale_ids:
            await self._redis.asyncio.srem_telemetry(SSH_CONNECTION_IDS_KEY, *stale_ids)
        return connections
