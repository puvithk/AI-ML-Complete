

from langgraph.graph import StateGraph , END , START
from .state import WebSearchResultState
from .nodes import clean_web_search , web_search_node


builder = StateGraph(WebSearchResultState)
builder.add_node('web_search_node' , web_search_node)
builder.add_node('clean_web_search' , clean_web_search)


builder.add_edge(START ,"web_search_node" )
builder.add_edge('web_search_node' , 'clean_web_search')
builder.add_edge('clean_web_search' , END)
web_search_subagent = builder.compile()

