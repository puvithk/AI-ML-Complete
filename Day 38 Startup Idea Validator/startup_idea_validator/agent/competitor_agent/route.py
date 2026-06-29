from typing import Literal
from .state import CompetitorState , WebSearchResultState
from langgraph.types import Send

def competitor_router(state: CompetitorState,) -> Literal["web_scraper", "web_search", "draft_report"]:

    return state['decision']



def route_questions(state : CompetitorState):
    return [
        Send(
            "web_search_node" ,
            {
          "question" :  question
            }
             
             ) for question in state['questions']
    ]