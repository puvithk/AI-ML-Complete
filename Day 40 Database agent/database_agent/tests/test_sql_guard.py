"""Tests for the read-only SQL guard - the primary write-protection layer."""

import pytest

from utils.sql_guard import UnsafeSQLError, is_safe_identifier, validate_read_only


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "select id, total from orders where total > 100 limit 5",
    "WITH t AS (SELECT * FROM orders) SELECT * FROM t",
    "  SELECT 1;  ",  # trailing semicolon + whitespace is fine
    "(SELECT id FROM orders)",
])
def test_allows_read_only_queries(sql):
    assert validate_read_only(sql)  # does not raise


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE orders SET total = 0",
    "INSERT INTO orders VALUES (1)",
    "DROP TABLE orders",
    "TRUNCATE orders",
    "ALTER TABLE orders ADD COLUMN x int",
    "COPY orders TO '/tmp/x.csv'",
    "SELECT 1; DELETE FROM orders",         # stacked statements
    "SELECT 1; SELECT 2",                    # multiple statements
    "CREATE TABLE t AS SELECT * FROM orders",
    "SET statement_timeout = 0",
    "",
    "   ",
])
def test_rejects_writes_and_multistatements(sql):
    with pytest.raises(UnsafeSQLError):
        validate_read_only(sql)


@pytest.mark.parametrize("sql", [
    "DESCRIBE orders",
    "DESC orders",
    "SHOW TABLES",
    "EXPLAIN SELECT * FROM orders",
])
def test_allows_readonly_introspection(sql):
    assert validate_read_only(sql)


def test_keyword_inside_string_literal_is_allowed():
    # 'delete' appears only inside a string literal -> still a read-only query.
    assert validate_read_only("SELECT * FROM orders WHERE note = 'please delete me'")


def test_keyword_inside_comment_is_stripped():
    assert validate_read_only("SELECT 1 -- drop table orders\n")
    assert validate_read_only("SELECT 1 /* update orders */")


def test_column_named_like_keyword_is_fine():
    # 'created_at' must not trip the CREATE check (word boundaries).
    assert validate_read_only("SELECT created_at, updated_at FROM orders")


@pytest.mark.parametrize("name,ok", [
    ("orders", True),
    ("ts_order_sps", True),
    ("_tmp1", True),
    ("orders; DROP TABLE x", False),
    ("orders WHERE 1=1", False),
    ("", False),
    ("1orders", False),
])
def test_is_safe_identifier(name, ok):
    assert is_safe_identifier(name) is ok
