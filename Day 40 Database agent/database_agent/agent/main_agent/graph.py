from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from .state import OrchestratorAgentState
from .nodes import question_decomposer , merge , final_result
from .route import route_to , critic_route
from ..sql_agent.graph import agent as sql_agent
from ...utils.database import db_connection
from ...utils.checkpointer import get_checkpointer
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


def run_sql_agent(state, config: RunnableConfig):
    """Run the SQL subgraph for one decomposed question.

    The subgraph is invoked here instead of being added as a node directly,
    so that only `final_answer` (which has an `operator.add` reducer) is
    written back to the parent state. If the subgraph were a node, it would
    also write back `user_question`, and with several parallel branches (one
    per decomposed question) that produces multiple writes to the parent's
    no-reducer `user_question` channel -> InvalidUpdateError.

    Each branch borrows its own pooled DB connection and subgraph thread_id,
    since parallel branches must not share a single psycopg2 connection.
    """
    question = state["user_question"]
    with db_connection() as conn:
        sub_config = {
            "configurable": {
                "thread_id": f"sql:{abs(hash(question))}",
                "conn": conn,
            }
        }
        try:
            result = sql_agent.invoke(
                {"user_question": question, "retry_no": state.get("retry_no", 0)},
                config=sub_config,
            )
        except Exception as exc:  # noqa: BLE001 - one branch must not kill the run
            logger.exception("SQL sub-agent failed for question: %s", question)
            return {
                "final_answer": [
                    "Could not retrieve an answer for one part of the request."
                ]
            }

    final_answer = result.get("final_answer") or []
    if isinstance(final_answer, str):
        final_answer = [final_answer] if final_answer else []
    return {"final_answer": final_answer}


builder = StateGraph(OrchestratorAgentState)
builder.add_node("question_decomposer", question_decomposer)
builder.add_node("sql_agent", run_sql_agent)
builder.add_node("merger", merge)
builder.add_node("final_result", final_result)

builder.add_edge(START, 'question_decomposer')
builder.add_conditional_edges("question_decomposer", route_to)

# All parallel sql_agent branches fan in to the merger, which summarizes and
# then hands off to the critic. The critic either finalizes or asks for
# another decomposition cycle (bounded by MAX_REVISIONS in critic_route).
builder.add_edge("sql_agent", "merger")
builder.add_conditional_edges(
    "merger",
    critic_route,
    {
        "final_result": "final_result",
        "question_decomposer": "question_decomposer",
    },
)
builder.add_edge("final_result", END)

agent = builder.compile(checkpointer=get_checkpointer())
