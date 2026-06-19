from graph.state import State
from typing import Literal
from langgraph.graph import END


def should_continue_research(state:State) -> Literal['__end__' , 'research_agent']:
    # If iteration is more that 3 thenm return 
    if not state['need_more_research']:
        return '__end__'
    else :
        return 'research_agent'

