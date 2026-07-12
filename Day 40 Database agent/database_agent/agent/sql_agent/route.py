from ..sql_agent.state import SqlQueryRetrialState
from ...utils.config import get_settings


def retry_query(state: SqlQueryRetrialState) -> str:
    """Conditional edge after `critic`.

    NOTE: state is NOT mutated here. LangGraph silently discards writes made
    inside conditional-edge functions, so `retry_no` is incremented only in
    the `critic` node. This function purely routes.
    """
    max_retries = get_settings().sql_max_retries
    if state.get('verified'):
        return 'excute_query'
    if state.get('retry_no', 0) > max_retries:
        return 'format_result'
    return 'query_generator'


def is_query_successful(state: SqlQueryRetrialState) -> str:
    """Conditional edge after `execute_query`.

    A successful result moves on to summarization; a failure retries
    (bounded by retry_no) before giving up.
    """
    max_retries = get_settings().sql_max_retries
    if state.get('result') is not None:
        return 'format_result'
    if state.get('retry_no', 0) > max_retries:
        return 'format_result'
    return 'query_generator'
