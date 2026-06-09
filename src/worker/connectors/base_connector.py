"""Base connector class for external tool calling / data sources."""

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class ConnectorError(RuntimeError):
    """Raised when a connector operation fails in a controlled way."""


class ConnectorResult(TypedDict):
    """Standard envelope returned by :meth:`BaseConnector.execute`.

    ``data`` is connector-specific on success; ``error`` is a human-readable
    message on failure. ``metadata`` carries auxiliary fields (execution time,
    row count, materialised URI, etc.).
    """

    success: bool
    data: Any
    error: str | None
    metadata: dict[str, Any]


class BaseConnector(ABC):
    """Abstract base class for data connectors.

    Connectors provide a uniform interface for executors to interact with
    external systems (databases, APIs, vector stores, etc.) through tool calls
    or structured queries.
    """

    #: Human-readable identifier for logging/telemetry
    name: str = "base_connector"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the external system.

        This method is called before the first query/execute call. Subclasses
        should implement connection pooling, authentication, etc. here.

        Raises:
            ConnectorError: If connection fails
        """
        raise NotImplementedError

    def disconnect(self) -> None:
        """Close connection to the external system.

        Called during cleanup or teardown. Should gracefully close connections,
        release resources, etc.
        """
        pass

    @abstractmethod
    def execute(
        self, query: str | list[str], *args: Any, **kwargs: Any
    ) -> ConnectorResult:
        """Execute a query/command on the external system.

        Args:
            query: The query string (SQL, API endpoint, etc.)
            *args: Additional positional arguments
            **kwargs: Additional connector-specific options

        Returns:
            A dict containing:
                - success: bool indicating if operation succeeded
                - data: Query results (list of dicts for SELECT, affected rows for DML,
                        etc.)
                - error: Error message if success=False
                - metadata: Optional additional info (execution time, row count, etc.)

        Raises:
            ConnectorError: For expected failures (syntax errors, permission denied,
            etc.)
        """
        raise NotImplementedError

    def validate_query(self, query: str) -> bool:
        """Validate a query before execution (optional safety check).

        Args:
            query: The query string to validate

        Returns:
            True if query appears safe to execute, False otherwise
        """
        # Default: allow all queries (subclasses can override for safety)
        return True

    def get_schema(self, table_name: str | None = None) -> dict[str, Any]:
        """Retrieve schema information from the external system.

        Args:
            table_name: Optional specific table name to get schema for.
                       If None, returns all tables/views/schemas.
                       If provided, returns detailed schema for that table.

        Returns:
            Schema metadata (tables, columns, types, etc.) specific to the connector
        """
        return {}

    def estimate_query_cost(self, query: str, **kwargs) -> dict[str, Any]:
        """Estimate the cost of executing a query without actually executing it.

        Args:
            query: The query string to estimate

        Returns:
            Dict containing cost estimation metrics specific to the connector
        """
        return {}

    def __enter__(self):
        """Context manager support."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.disconnect()
        return False
