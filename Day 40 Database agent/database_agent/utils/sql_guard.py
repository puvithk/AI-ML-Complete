"""Read-only SQL guard.

Validates that an LLM-generated statement is a single read-only query BEFORE
it is ever sent to the database. This is the primary write-protection layer
(the DB session read-only flag and never-committing are defense in depth).
"""

from __future__ import annotations

import re

# Statements / keywords that must never reach the DB.
_FORBIDDEN = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "COPY", "PRAGMA", "ATTACH", "DETACH", "CALL", "GRANT", "REVOKE",
    "MERGE", "REPLACE", "EXPORT", "IMPORT", "INSTALL", "LOAD", "VACUUM",
    "SET", "RESET", "EXECUTE", "PREPARE", "COMMENT", "REINDEX", "CLUSTER",
]
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE
)

# Read-only statement prefixes we accept. SELECT/WITH are the query forms;
# DESCRIBE/SHOW/EXPLAIN are read-only metadata/introspection statements used
# by the schema-loading helpers.
_ALLOWED_PREFIXES = ("SELECT", "WITH", "DESCRIBE", "DESC ", "SHOW", "EXPLAIN")

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class UnsafeSQLError(ValueError):
    """Raised when a statement is not a safe, single read-only query."""


def _strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return sql


def validate_read_only(sql: str) -> str:
    """Return a cleaned single read-only statement or raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement.")

    cleaned = _strip_comments(sql).strip()
    # Drop a single trailing semicolon.
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    # Reject multiple statements (any remaining semicolon).
    if ";" in cleaned:
        raise UnsafeSQLError("Multiple SQL statements are not allowed.")

    # Blank out string literals so their contents can't trip keyword checks.
    scan = _STRING_LITERAL_RE.sub("''", cleaned)

    # Must be a read-only statement.
    first = scan.lstrip("( \t\r\n").upper()
    if not first.startswith(_ALLOWED_PREFIXES):
        raise UnsafeSQLError("Only read-only (SELECT/WITH/DESCRIBE/SHOW/EXPLAIN) statements are allowed.")

    match = _FORBIDDEN_RE.search(scan)
    if match:
        raise UnsafeSQLError(
            f"Forbidden keyword '{match.group(1).upper()}' in query."
        )

    return cleaned


def is_safe_identifier(name: str) -> bool:
    """True if `name` is a plain SQL identifier (no injection surface)."""
    return bool(name) and bool(_IDENTIFIER_RE.match(name))
