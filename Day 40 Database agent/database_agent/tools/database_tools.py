"""Database access tools used by the SQL agent.

All query execution goes through `execute_sql_query`, which validates that
the statement is read-only BEFORE running it and caps the number of rows
returned.
"""

from __future__ import annotations

from ..utils.config import get_settings
from ..utils.logging_config import get_logger
from ..utils.sql_guard import UnsafeSQLError, is_safe_identifier, validate_read_only

logger = get_logger(__name__)


def execute_sql_query(conn, query) -> dict:
    """Validate and execute a read-only SQL query.

    Returns a self-describing result so consumers (and the LLM) know what each
    value means:

        {"columns": ["col1", "col2", ...], "rows": [(v1, v2, ...), ...]}

    Args:
        conn: A psycopg2 connection (borrowed from the pool, read-only).
        query (str): The SQL query to execute.

    Raises:
        UnsafeSQLError: if the statement is not a single read-only query.
        RuntimeError: if execution fails.
    """
    safe_query = validate_read_only(query)  # raises UnsafeSQLError on write ops
    max_rows = get_settings().max_result_rows

    with conn.cursor() as cursor:
        try:
            cursor.execute(safe_query)
            if cursor.description is None:
                # A read-only query always yields a result set; anything else
                # is unexpected - refuse and roll back.
                conn.rollback()
                raise UnsafeSQLError("Query did not return a result set.")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(max_rows)
            if cursor.rowcount is not None and cursor.rowcount > max_rows:
                logger.info(
                    "Result truncated to %s rows (query returned more).", max_rows
                )
            return {"columns": columns, "rows": rows}
        except UnsafeSQLError:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.warning("Query execution failed: %s", e)
            raise RuntimeError(f"Failed to execute SQL: {e}") from e


def get_database_schema(conn) -> list[str]:
    """Return the list of table names in the configured schema."""
    schema = get_settings().db_schema
    query = (
        "SELECT table_name FROM duckdb_tables() "
        f"WHERE schema_name = '{schema}' ORDER BY table_name;"
    )
    result = execute_sql_query(conn, query)
    rows = result.get("rows", [])
    return [row[0] for row in rows] if rows else []


def get_table_details_by_name(conn, table_name: str) -> dict:
    """Return column details for a table, validated against the live schema.

    `table_name` comes from the LLM, so it is checked to be a plain identifier
    AND to actually exist in the schema before being interpolated into a
    DESCRIBE statement (prevents SQL injection via table names).
    """
    if not is_safe_identifier(table_name):
        logger.warning("Rejected unsafe table identifier: %r", table_name)
        return {"table_name": table_name, "columns": []}

    known_tables = get_database_schema(conn)
    if table_name not in known_tables:
        logger.warning("Requested unknown table: %r", table_name)
        return {"table_name": table_name, "columns": []}

    result = execute_sql_query(conn, f"DESCRIBE {table_name};")
    rows = result.get("rows", [])
    if rows:
        return {
            "table_name": table_name,
            # (column_name, column_type) pairs, order preserved.
            "columns": [
                {"name": row[0], "type": row[1] if len(row) > 1 else None}
                for row in rows
            ],
        }
    return {"table_name": table_name, "columns": []}
