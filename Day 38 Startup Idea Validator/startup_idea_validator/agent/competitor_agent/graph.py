#Import libaries 
from langgraph.graph import StateGraph , START , END
from utils.llm import llm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from .state import CompetitorState
from .nodes import competitor_decision_engine , question_decompose  , merger , draft_report
from .route import competitor_router , route_questions
from .search_subgraph.graph import web_search_subagent
from tools.web_tools import web_search_tool , web_scraping_tool
#Compitator agent 
builder = StateGraph(CompetitorState)
#Basically researchs about the compittator present in the industies of the topic 
# Compitator decision engine routes to various nodes like  web search , web scraping , draft reporting
builder.add_node('competitor_decision_engine' , competitor_decision_engine)
builder.add_node('question_decomposer' , question_decompose)

builder.add_node('web_search_node' ,web_search_subagent )
builder.add_node('merger' , merger)
builder.add_node('draft_report' , draft_report)
builder.add_edge(START , 'competitor_decision_engine')
builder.add_conditional_edges('competitor_decision_engine' ,
                              competitor_router , {
                                  "web_scraper": END ,
                                "web_search": "question_decomposer",
                                  "draft_report" : "draft_report"
                              })
builder.add_conditional_edges('question_decomposer' ,
                              route_questions,
                              {
                                  "web_search" : 'web_search_node'
                              }
                              )
builder.add_edge('web_search_node' , 'merger')
builder.add_edge('merger' , 'competitor_decision_engine')


memory = InMemorySaver()
agent = builder.compile( checkpointer=memory)
# Search the similar companies (web scraping)
# Based on the result and website 
# Make a deep research (web scraping)
# And make the requeired draft report 
# Save the draft report 
# Critic engine to check the report and resource 
# Make sure that the details are relavant and accurate 

# Update the final report with resources taged 