"""Connectors for external systems (databases, APIs, etc.)."""

from .agent_connector import AgentConnector
from .base_connector import BaseConnector, ConnectorError
from .postgresql_connector import PostgreSQLConnector
from .s3_connector import S3Connector

__all__ = [
    "AgentConnector",
    "BaseConnector",
    "ConnectorError",
    "PostgreSQLConnector",
    "S3Connector",
]


def get_connector_from_spec(connection_string: str, **kwargs) -> BaseConnector:
    """Factory method to create connector instances from specification.

    Args:
        connection_string: Connection string with scheme prefix
        **kwargs: Additional connector-specific options (e.g., cert_data, use_ssl)
    """
    if connection_string.startswith("postgresql://"):
        return PostgreSQLConnector(connection_string=connection_string)
    elif connection_string.startswith("s3://"):
        return S3Connector(connection_string=connection_string, **kwargs)
    else:
        raise ConnectorError(f"Unsupported connector type: {connection_string}")
