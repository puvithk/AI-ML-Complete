from .state import CompetitorState , CompetitorDecision
from typing import Literal
from utils.llm import llm
from .prompt import compitator_decision_engine
def competitor_decision_engine(state : CompetitorState) -> CompetitorState :
    # BAsed on the prevoius data and info source we need to update the state 
    llm_structure = llm.with_structured_output(CompetitorDecision)


    prompt = compitator_decision_engine.format(pitch_summary=state['pitch_summary'], source=state['sources'])
    # Use web scraper or web search or to draft research agent

    response = llm_structure.invoke(prompt)
    data = response.model_dump()
    state['decision_feedback'] = data['decision_feedback']
    state['decision'] = data['decision']
    return state


