from database_agent.agent.main_agent.state import OrchestratorAgentState, DecomposerResult, FinalFormatedResult , ChitChatResponse , RewriteAndRoute
from database_agent.agent.main_agent.prompt import question_decomposer_prompt, final_answer_prompt , chit_chat_prompt, REWRITE_AND_ROUTE_PROMPT
from database_agent.utils.llm import get_llm
from database_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


def question_decomposer(state: OrchestratorAgentState):
    llm_structure = get_llm().with_structured_output(DecomposerResult)

    prompt = question_decomposer_prompt.format(user_question=state['user_question'])

    response = llm_structure.invoke(prompt)
    questions = response.model_dump()['questions']
    logger.info("Decomposed question into %d sub-question(s).", len(questions))
    return {
        "decomposed_question": questions
    }

def chit_chat(state : OrchestratorAgentState) -> OrchestratorAgentState:
    """This node returns only normal talks like hello , hi , how r u  """
    llm_structure = get_llm().with_structured_output(ChitChatResponse)

    prompt = chit_chat_prompt.format(user_question=state['user_question'])
    response = llm_structure.invoke(prompt)

    return {
        "final_answer" : response
    }


def rewrite_and_route(state : OrchestratorAgentState) -> OrchestratorAgentState:
    llm_structure = get_llm().with_structured_output(RewriteAndRoute)

    prompt = REWRITE_AND_ROUTE_PROMPT.format(user_question=state['user_question'] , working_summary=state['working_summary'])

    response = llm_structure.invoke(prompt)

    return {
        "user_question" : response.model_dump()['rewrittern_query'] , 
        "next_route" : response.model_dump()['next_route']
    }






def merge(state: OrchestratorAgentState):
    """Summarize the SQL agent results and bump the revision counter.

    `revision_count` bounds the critic -> question_decomposer retry loop so it
    can never run forever.
    """
    llm_structure = get_llm().with_structured_output(FinalFormatedResult)

    prompt = final_answer_prompt.format(
        user_question=state['user_question'],
        agent_result=state.get('final_answer', []),
    )

    response = llm_structure.invoke(prompt)
    return {
        "summarized_output": response.model_dump()['final_answer'],
        "revision_count": state.get('revision_count', 0) + 1,
    }


def final_result(state: OrchestratorAgentState):
    return state
