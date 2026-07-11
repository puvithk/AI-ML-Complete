from agent.sql_agent.state import SqlQueryRetrialState, RequiredTables, QueryResult, CriticResult, FinalAnswer
from agent.sql_agent.prompt import get_required_table_prompt, generate_sql_query_prompt, critic_check_prompt, final_answer_prompt

from tools.database_tools import get_database_schema, execute_sql_query, get_table_details_by_name
from langchain_core.runnables import RunnableConfig

from utils.llm import get_llm
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _conn(config: RunnableConfig):
    """Pull the DB connection out of the graph config, or fail clearly."""
    conn = (config or {}).get("configurable", {}).get("conn")
    if conn is None:
        raise RuntimeError("No DB connection supplied in graph config['configurable']['conn'].")
    return conn


# Node to get the list of tables in the database.
def get_database_details(state: SqlQueryRetrialState, config: RunnableConfig) -> dict:
    return {
        'table_details': get_database_schema(_conn(config)),
    }


def get_required_tables(state: SqlQueryRetrialState) -> dict:
    """Ask the LLM which tables are needed to answer the question."""
    llm_structured = get_llm().with_structured_output(RequiredTables)

    prompt = get_required_table_prompt.format(
        user_question=state['user_question'],
        tables_available=state['table_details'],
    )
    response = llm_structured.invoke(prompt)

    return {
        'required_tables': response.model_dump()['table_name'],
    }


# Node to get the details of the relevant tables.
def get_table_details(state: SqlQueryRetrialState, config: RunnableConfig):
    conn = _conn(config)
    table_details = {}
    for table in state['required_tables']:
        table_details[table] = get_table_details_by_name(conn, table).get("columns")
        logger.debug("Loaded schema for table %s", table)
    return {
        'table_details': table_details
    }


# Node to generate the SQL query.
def get_query(state: SqlQueryRetrialState):
    llm_structed = get_llm().with_structured_output(QueryResult)
    prev_query = state.get('last_sql', "")
    prev_error = state.get('last_error', "")
    prev_suggestion = state.get('last_suggetion', "")
    prev_issues = state.get('last_issues', "")

    prompt = generate_sql_query_prompt.format(
        user_question=state['user_question'],
        relevant_tables=state['required_tables'],
        table_details=state['table_details'],
        prev_query=prev_query,
        prev_error=prev_error,
        prev_suggetion=prev_suggestion,
        prev_issues=prev_issues,
    )
    response = llm_structed.invoke(prompt)
    result = response.model_dump()
    return {
        'query': result['query'],
        'confidence': result['confidence'],
        'last_sql': result['query'],
    }


# Node validates the query before it is executed.
def critic(state: SqlQueryRetrialState) -> dict:
    """
    Validate the generated query before it's executed.

    retry_no is incremented HERE (inside a real node) instead of inside the
    conditional-edge function. Conditional-edge functions in LangGraph only
    return a routing string - any state mutation inside them is silently
    discarded, so incrementing retry_no there never actually persisted.
    """
    llm_structured = get_llm().with_structured_output(CriticResult)
    prompt = critic_check_prompt.format(
        user_question=state['user_question'],
        table_details=state['table_details'],
        sql_query=state['query'],
    )

    response = None
    for attempt in range(3):  # bounded instead of the original `while True`
        try:
            response = llm_structured.invoke(prompt)
            if response is not None:
                break
        except Exception as e:  # noqa: BLE001
            logger.warning("critic invoke failed (attempt %d/3): %s", attempt + 1, e)

    retry_no = state.get('retry_no', 0) + 1

    if response is None:
        # Structured output kept failing - fail safe instead of hanging forever.
        return {
            'verified': False,
            'confidence': 0,
            'last_issues': "Critic failed to produce a response after 3 attempts.",
            'last_suggetion': "Retry query generation.",
            'summary': "",
            'retry_no': retry_no,
        }

    result = response.model_dump()
    return {
        'verified': result['approved'],
        'confidence': result['confidence'],
        'last_issues': result['issues'],
        'last_suggetion': result['suggestions'],
        'summary': result['summary'],
        'retry_no': retry_no,
    }


# Node executes the query and gets the result.
def execute_query(state: SqlQueryRetrialState, config: RunnableConfig):
    try:
        result = execute_sql_query(_conn(config), state['query'])
    except Exception as e:  # noqa: BLE001 - captured for the retry loop
        logger.warning("execute_query failed: %s", e)
        return {
            'result': None,
            'last_error': str(e),
            'verified': False,
        }
    return {'result': result}


# Node turns the raw result into a natural-language answer.
def format_result(state: SqlQueryRetrialState):
    result = state.get('result')
    # `result` is {"columns": [...], "rows": [...]}; no rows means no data.
    rows = result.get('rows') if isinstance(result, dict) else result
    if not rows:
        return {
            'final_answer': ["No matching records were found for this question."]
        }

    llm_structed = get_llm().with_structured_output(FinalAnswer)
    prompt = final_answer_prompt.format(
        user_question=state['user_question'],
        query_results=result,  # includes column names so the LLM can label values
    )
    response = llm_structed.invoke(prompt)
    return {
        'final_answer': [response.model_dump()['answer']]
    }
