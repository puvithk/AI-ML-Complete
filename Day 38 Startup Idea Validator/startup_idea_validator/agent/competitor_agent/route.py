from typing import Literal
from .state import CompetitorState


def competitor_router(state: CompetitorState,) -> Literal["web_scraper", "web_search", "draft_report"]:

    return state['decision']
