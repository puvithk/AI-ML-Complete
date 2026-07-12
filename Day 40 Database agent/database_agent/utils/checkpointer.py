"""Checkpointer factory.

Returns a durable, file-backed SQLite checkpointer when `CHECKPOINT_DB` is
set (survives restarts), otherwise an in-memory checkpointer. Both graphs
share one instance so a session's state is consistent.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver

from ..utils.config import get_settings
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer():
    path = get_settings().checkpoint_db.strip()
    if not path:
        logger.info("Using in-memory checkpointer (set CHECKPOINT_DB for durability).")
        return InMemorySaver()

    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # check_same_thread=False so the connection can be shared across the
        # threaded server workers.
        conn = sqlite3.connect(path, check_same_thread=False)
        logger.info("Using SQLite checkpointer at %s", path)
        return SqliteSaver(conn)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to init SQLite checkpointer (%s); falling back to in-memory.",
            exc,
        )
        return InMemorySaver()
