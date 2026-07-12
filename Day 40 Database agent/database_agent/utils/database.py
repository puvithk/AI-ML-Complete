"""Database connectivity: pooled, read-only, timeout-bounded.

Primary API is the `db_connection()` context manager, which borrows a
connection from a process-wide pool, prepares it as read-only, yields it,
and always returns it to the pool (rolling back so nothing is ever
committed).

    from utils.database import db_connection
    with db_connection() as conn:
        ...
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool as pg_pool

from ..utils.config import get_settings
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Lazily build and return the shared threaded connection pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                s = get_settings()
                logger.info(
                    "Creating DB pool (min=%s max=%s host=%s db=%s)",
                    s.db_pool_min, s.db_pool_max, s.db_host, s.db_name,
                )
                _pool = pg_pool.ThreadedConnectionPool(
                    minconn=s.db_pool_min,
                    maxconn=s.db_pool_max,
                    host=s.db_host,
                    port=s.db_port,
                    user=s.db_user,
                    password=s.db_password,
                    dbname=s.db_name,
                )
    return _pool


def _prepare(conn) -> None:
    """Best-effort: make the session read-only and time-bounded.

    Both statements are engine-specific (this is DuckDB over the Postgres
    wire), so failures are tolerated - the SQL validator and the never-commit
    policy are the primary safeguards.
    """
    s = get_settings()
    with conn.cursor() as cur:
        if s.db_statement_timeout_ms > 0:
            try:
                cur.execute(f"SET statement_timeout = {int(s.db_statement_timeout_ms)}")
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                logger.debug("statement_timeout not supported: %s", exc)
        try:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            logger.debug("read-only session not supported: %s", exc)
    conn.rollback()


@contextmanager
def db_connection() -> Iterator["psycopg2.extensions.connection"]:
    """Borrow a prepared connection from the pool and always return it."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        _prepare(conn)
        yield conn
    finally:
        try:
            conn.rollback()  # never commit - this is a read-only agent
        except Exception:  # noqa: BLE001
            pass
        pool.putconn(conn)


def get_conn():
    """Backward-compatible single connection borrow.

    Prefer `db_connection()`. Callers of `get_conn()` MUST return the
    connection with `release_conn()` to avoid pool exhaustion.
    """
    conn = get_pool().getconn()
    _prepare(conn)
    return conn


def release_conn(conn) -> None:
    if conn is None:
        return
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        pass
    try:
        get_pool().putconn(conn)
    except Exception:  # noqa: BLE001
        pass


def check_health() -> bool:
    """Return True if a trivial query succeeds against the DB."""
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB health check failed: %s", exc)
        return False


def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.closeall()
        finally:
            _pool = None
