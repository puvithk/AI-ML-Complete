from .state import OrchestratorAgentState, CriticRouteState

from langgraph.types import Send
from .prompt import critic_prompt

from database_agent.utils.llm import get_llm 
from database_agent.utils.config import get_settings
from database_agent.utils.logging_config  import get_logger


logger = get_logger(__name__)


def route_to(state: OrchestratorAgentState):
    return [
        Send(
            'sql_agent',
            {
                "user_question": questions,
                "retry_no": 0,
            }
        )
        for questions in state['decomposed_question']
    ]


def critic_route(state: OrchestratorAgentState):
    """Decide whether the answer is complete or another retrieval cycle is
    needed.

    Hard-bounded by MAX_REVISIONS: once the loop has run that many times we
    always finish, so a critic that keeps asking for "more" can never spin
    forever.
    """
    revisions = state.get('revision_count', 0)
    max_revisions = get_settings().max_revisions
    if revisions > max_revisions:
        logger.info("Revision cap (%d) reached; finalizing.", max_revisions)
        return "final_result"

    llm_with_structured = get_llm().with_structured_output(CriticRouteState)

    prompt = critic_prompt.format(
        user_question=state['user_question'],
        decomposed_question=state.get('decomposed_question', []),
        final_answer=state.get('final_answer', []),
    )

    response = llm_with_structured.invoke(prompt)
    decision = response.model_dump()['next_node']
    logger.info("Critic decision: %s (revision %d).", decision, revisions)
    return decision


