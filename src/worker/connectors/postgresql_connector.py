"""PostgreSQL connector using psycopg3."""

import logging
import time
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import tuple_row
from psycopg.sql import SQL, Identifier, Literal

from .base_connector import BaseConnector, ConnectorError, ConnectorResult

logger = logging.getLogger(__name__)


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL connector using psycopg3.

    Configuration keys:
        - connection_string: PostgreSQL connection string (required)
          Example: "postgresql://user:pass@host:port/dbname"
        - read_only: Enforce read-only mode (default: True)
          When True, only SELECT queries are allowed

    Example config:
        {
            "connection_string": "postgresql://user:pass@localhost:5432/mydb",
            "read_only": true
        }
    """

    name = "postgresql"

    def __init__(self, connection_string: str, read_only: bool = True):

        self._connection: psycopg.Connection | None = None
        self._connection_string = connection_string
        # Default to read-only mode for safety
        self._read_only = read_only

    def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        if self._connection is not None:
            return  # Already connected

        try:
            self._connection = psycopg.connect(
                self._connection_string, row_factory=tuple_row, autocommit=True
            )
            logger.info("Connected to PostgreSQL: %s", self._connection.info.dbname)
        except Exception as e:
            raise ConnectorError(f"Failed to connect to PostgreSQL: {e}") from e

    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection is not None:
            try:
                self._connection.close()
                logger.info("Disconnected from PostgreSQL")
            except Exception as e:
                logger.warning("Error closing PostgreSQL connection: %s", e)
            finally:
                self._connection = None

    def execute(
        self,
        query: str | list[str],
        params_identifier: dict[str, str] | None = None,
        params_literal: dict[str, Any] | None = None,
        fetch_size: int | None = None,
        **_,
    ) -> ConnectorResult:
        """Execute a SQL query on PostgreSQL.

        Args:
            query: SQL query string (can use %(name)s placeholders for params)
            params_identifier: Optional dict of identifier parameters for SQL formatting
            params_literal: Optional dict of literal parameters for SQL formatting
            - fetch_size: Number of rows to fetch for SELECT (default: all)

        Returns:
            Dict with:
                - success: bool
                - data: List of dicts for SELECT queries, or None for DML/DDL
                - error: Error message if failed
                - metadata: Query execution time, row count, etc.
        """
        if isinstance(query, list):
            raise NotImplementedError("Batch queries not supported in this executor.")
        params_identifier = params_identifier or {}
        params_literal = params_literal or {}

        if self._connection is None:
            self.connect()
            assert self._connection is not None

        start_time = time.time()
        result: ConnectorResult = {
            "success": False,
            "data": None,
            "error": None,
            "metadata": {},
        }
        formatted_query: str | None = None

        try:
            with self._connection.cursor() as cursor:
                logger.debug(
                    "Executing PostgreSQL query: %s, params_identifier=%s, "
                    "params_literal=%s",
                    query,
                    params_identifier,
                    params_literal,
                )
                query_formatted = SQL(query).format(  # type: ignore
                    **{
                        key: Identifier(value)
                        for key, value in params_identifier.items()
                    },
                    **{key: Literal(value) for key, value in params_literal.items()},
                )
                formatted_query = query_formatted.as_string(cursor)
                logger.debug("Formatted query: %s", formatted_query)

                # Execute query with optional parameters
                cursor.execute(query_formatted)

                # Determine query type and fetch results accordingly
                query_upper = query.strip().upper()

                if query_upper.startswith(("SELECT", "WITH")):
                    # Fetch results for SELECT/CTE queries
                    if fetch_size:
                        data = cursor.fetchmany(fetch_size)
                    else:
                        data = cursor.fetchall()
                    assert cursor.description is not None
                    header = [desc.name for desc in cursor.description]
                    df = pd.DataFrame(data, columns=header)
                    if len(df) == 0:
                        logger.warning(
                            "Query returned an empty result set. Query: %s",
                            formatted_query,
                        )
                    result["data"] = df
                else:
                    raise NotImplementedError(
                        "Only SELECT/CTE queries are supported in this executor."
                    )

                result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            logger.error("PostgreSQL query failed: %s", e, exc_info=True)
            raise ConnectorError(f"Query execution failed: {e}") from e

        finally:
            execution_time = time.time() - start_time
            result["metadata"] = {
                "execution_time_sec": execution_time,
                "query_length": len(query),
            }

        return result

    def validate_query(self, query: str) -> bool:
        """Validate SQL query for basic safety checks.

        Args:
            query: SQL query to validate

        Returns:
            True if query passes basic validation, False otherwise
        """
        query_upper = query.strip().upper()

        # Basic safety: disallow multiple statements (prevent injection)
        if ";" in query.strip()[:-1]:  # Allow trailing semicolon
            logger.warning("Query contains multiple statements, rejecting for safety")
            return False

        # If read-only mode, only allow SELECT and WITH (CTE) queries
        if self._read_only:
            if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
                logger.warning(
                    "Read-only mode enabled: only SELECT and WITH queries allowed, "
                    "got: %s",
                    query_upper[:50],
                )
                return False

            # Check for data modification keywords even in SELECT queries
            # (e.g., SELECT with subqueries that modify data)
            modification_keywords = [
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "CREATE",
                "ALTER",
                "TRUNCATE",
                "GRANT",
                "REVOKE",
                "EXEC",
                "EXECUTE",
            ]
            for keyword in modification_keywords:
                if keyword in query_upper:
                    logger.warning(
                        "Read-only mode: query contains modification keyword '%s', "
                        "rejecting",
                        keyword,
                    )
                    return False
        else:
            # Non-read-only mode: still block extremely dangerous operations
            dangerous_keywords = [
                "DROP DATABASE",
                "DROP SCHEMA",
                "TRUNCATE",
                "ALTER SYSTEM",
            ]
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    logger.warning(
                        "Query contains dangerous keyword '%s', rejecting", keyword
                    )
                    return False

        return True

    def get_schema(self, table_name: str | None = None) -> dict[str, Any]:
        """Retrieve PostgreSQL schema information.

        Args:
            table_name: Optional specific table name to get schema for.
                       Can be "schema.table" or just "table" (searches all schemas).
                       If None, returns all tables, views, and schemas.

        Returns:
            Dict containing:
                - If table_name is None:
                    - tables: List of table names
                    - views: List of view names
                    - schemas: List of schema names
                - If table_name is provided:
                    - table: Table name
                    - schema: Schema name
                    - columns: List of dicts with column info (name, type, nullable,
                               default)
        """
        if self._connection is None:
            self.connect()
            assert self._connection is not None

        # If specific table requested, return detailed column information
        if table_name:
            return self._get_table_schema(table_name)

        # Otherwise return list of all tables, views, and schemas
        schema_info: dict[str, list[str]] = {"tables": [], "views": [], "schemas": []}

        try:
            # Get all schemas
            with self._connection.cursor() as cursor:
                cursor.execute("""
                    SELECT schema_name 
                    FROM information_schema.schemata
                    WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                """)
                schema_info["schemas"] = [
                    row["schema_name"] for row in cursor.fetchall()  # type: ignore
                ]

            # Get all tables
            with self._connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                    AND table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name
                """)
                schema_info["tables"] = [
                    f"{row['table_schema']}.{row['table_name']}"  # type: ignore
                    for row in cursor.fetchall()
                ]

            # Get all views
            with self._connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.views
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name
                """)
                schema_info["views"] = [
                    f"{schema}.{table}" for schema, table in cursor.fetchall()
                ]

        except Exception as e:
            logger.error("Failed to retrieve schema: %s", e)
            raise ConnectorError(f"Failed to retrieve schema: {e}") from e

        return schema_info

    def _get_table_schema(self, table_name: str) -> dict[str, Any]:
        """Get detailed schema for a specific table.

        Args:
            table_name: Table name (can be "schema.table" or just "table")

        Returns:
            Dict with table schema details including columns
        """
        # Parse schema and table name
        if "." in table_name:
            schema_name, tbl_name = table_name.split(".", 1)
        else:
            schema_name = None
            tbl_name = table_name

        assert self._connection is not None
        try:
            with self._connection.cursor() as cursor:
                # Query for column information
                if schema_name:
                    cursor.execute(
                        """
                        SELECT 
                            c.column_name,
                            c.data_type,
                            c.is_nullable,
                            c.column_default,
                            c.character_maximum_length,
                            c.numeric_precision,
                            c.numeric_scale
                        FROM information_schema.columns c
                        WHERE c.table_schema = %(schema)s
                        AND c.table_name = %(table)s
                        ORDER BY c.ordinal_position
                    """,
                        {"schema": schema_name, "table": tbl_name},
                    )
                else:
                    # Search all schemas (except system schemas)
                    cursor.execute(
                        """
                        SELECT 
                            c.table_schema,
                            c.column_name,
                            c.data_type,
                            c.is_nullable,
                            c.column_default,
                            c.character_maximum_length,
                            c.numeric_precision,
                            c.numeric_scale
                        FROM information_schema.columns c
                        WHERE c.table_name = %(table)s
                        AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY c.ordinal_position
                    """,
                        {"table": tbl_name},
                    )

                columns = cursor.fetchall()

                if not columns:
                    raise ConnectorError(f"Table '{table_name}' not found")

                # Determine row shape: with schema provided query returns 7 cols,
                # without schema returns 8 cols (first is table_schema).
                first_row = columns[0]
                has_table_schema_col = len(first_row) == 8

                # If no schema specified and multiple found, use first one
                if not schema_name and has_table_schema_col:
                    schema_name = first_row[0] or "public"
                elif not schema_name:
                    schema_name = "public"

                result: dict[str, Any] = {
                    "table": tbl_name,
                    "schema": schema_name,
                    "columns": [],
                }

                for row in columns:
                    if has_table_schema_col:
                        # row => (table_schema, column_name, data_type, is_nullable,
                        #         column_default, character_maximum_length,
                        #         numeric_precision, numeric_scale)
                        (
                            _,
                            col_name,
                            col_type,
                            is_nullable,
                            default,
                            max_len,
                            precision,
                            scale,
                        ) = row
                    else:
                        # row => (column_name, data_type, is_nullable, column_default,
                        #         character_maximum_length, numeric_precision,
                        #         numeric_scale)
                        (
                            col_name,
                            col_type,
                            is_nullable,
                            default,
                            max_len,
                            precision,
                            scale,
                        ) = row

                    result["columns"].append(
                        {
                            "name": col_name,
                            "type": col_type,
                            "nullable": (is_nullable == "YES"),
                            "default": default,
                            "max_length": max_len,
                            "precision": precision,
                            "scale": scale,
                        }
                    )

                return result

        except Exception as e:
            logger.error("Failed to retrieve table schema for '%s': %s", table_name, e)
            raise ConnectorError(f"Failed to retrieve table schema: {e}") from e

    def estimate_query_cost(
        self,
        query: str,
        params_identifier: dict[str, str] | None = None,
        params_literal: dict[str, Any] | None = None,
        **_,
    ) -> dict[str, Any]:
        """Estimate query cost using EXPLAIN without executing the query.

        Args:
            query: SQL query to estimate
            params_identifier: Optional dict of identifier parameters for SQL formatting
            params_literal: Optional dict of literal parameters for SQL formatting

        Returns:
            Dict with:
                - estimated_cost: Query cost estimate (arbitrary units)
                - estimated_rows: Estimated number of rows
                - planning_time_ms: Time spent planning the query
                - execution_plan: Full EXPLAIN output
                - error: Error message if estimation failed
        """
        if self._connection is None:
            self.connect()
            assert self._connection is not None

        result: dict[str, Any] = {
            "estimated_cost": None,
            "estimated_rows": None,
            "planning_time_ms": None,
            "execution_plan": None,
            "error": None,
        }

        params_identifier = params_identifier or {}
        params_literal = params_literal or {}

        try:
            with self._connection.cursor() as cursor:
                # Build EXPLAIN query
                explain_query = f"EXPLAIN (FORMAT JSON, VERBOSE) {query}"

                # Format with parameters
                query_formatted = SQL(explain_query).format(  # type: ignore
                    **{
                        key: Identifier(value)
                        for key, value in params_identifier.items()
                    },
                    **{key: Literal(value) for key, value in params_literal.items()},
                )

                logger.debug("Executing EXPLAIN: %s", query_formatted.as_string(cursor))
                cursor.execute(query_formatted)

                explain_result = cursor.fetchone()
                if explain_result and len(explain_result) > 0:
                    plan_json = explain_result[0]

                    if isinstance(plan_json, list) and len(plan_json) > 0:
                        plan = plan_json[0]

                        # Extract cost and row estimates
                        if "Plan" in plan:
                            plan_node = plan["Plan"]
                            result["estimated_cost"] = plan_node.get("Total Cost")
                            result["estimated_rows"] = plan_node.get("Plan Rows")

                        # Extract planning time
                        result["planning_time_ms"] = plan.get("Planning Time")
                        result["execution_plan"] = plan_json

                        logger.debug(
                            "Cost estimate: %.2f, rows: %s",
                            result["estimated_cost"] or 0,
                            result["estimated_rows"] or "?",
                        )

        except Exception as e:
            result["error"] = str(e)
            logger.error("Failed to estimate query cost: %s", e, exc_info=True)

        return result
