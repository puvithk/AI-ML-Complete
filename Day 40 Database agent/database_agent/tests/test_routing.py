"""Tests for the SQL agent's pure routing functions."""

from agent.sql_agent.route import is_query_successful, retry_query
from utils.config import get_settings

MAX = get_settings().sql_max_retries


def test_verified_query_executes():
    assert retry_query({"verified": True, "retry_no": 0}) == "excute_query"


def test_unverified_retries_until_cap():
    assert retry_query({"verified": False, "retry_no": 0}) == "query_generator"
    assert retry_query({"verified": False, "retry_no": MAX + 1}) == "format_result"


def test_missing_keys_default_safely():
    # No 'verified'/'retry_no' keys -> should not raise, routes to regenerate.
    assert retry_query({}) == "query_generator"


def test_successful_result_formats():
    assert is_query_successful({"result": [(1, 2)]}) == "format_result"


def test_failed_result_retries_then_gives_up():
    assert is_query_successful({"result": None, "retry_no": 0}) == "query_generator"
    assert is_query_successful({"result": None, "retry_no": MAX + 1}) == "format_result"


def test_is_query_successful_missing_keys():
    assert is_query_successful({}) == "query_generator"
