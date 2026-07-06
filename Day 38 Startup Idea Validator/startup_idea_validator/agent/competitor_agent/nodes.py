from .state import CompetitorState , CompetitorDecision , QuestionDecomposed  , DraftReportResult
from typing import Literal
from utils.llm import llm
from .prompt import compitator_decision_engine , question_decomposer_engine  , draft_report_prompt

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

def question_decompose(state : CompetitorState):
    #Take the pitch summary and the feedback by the user to provide the alll    the question which should be searched  
    # Create a llm with structure 

    llm_structure = llm.with_structured_output(QuestionDecomposed)

    # Create a prompt 

    prompt = question_decomposer_engine.format(pitch_summary=state['pitch_summary'] , decision_feedback  = state['decision_feedback'])


    response = llm_structure.invoke(prompt)

    return {
        "questions" : response.model_dump()['questions']
    }

def merger(state : CompetitorState):

    print("All data is mergered : ")
    return state


def draft_report(state : CompetitorState):
    llm_with_structure = llm.with_structured_output(DraftReportResult)


    prompt = draft_report_prompt.format(pitch_text=state['pitch_text'] , pitch_summary=state['pitch_summary'] , sources=state['sources'])


    response = llm_with_structure.invoke(prompt)


    return {
        "report" : response.model_dump()['report']
    }

