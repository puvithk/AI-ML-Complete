#Import libaries 
from langgraph.graph import StateGraph , START , END
from utils.llm import llm
from langgraph.checkpoint.memory import InMemorySaver
from .state import CompetitorState
from .nodes import competitor_decision_engine
from .route import competitor_router
#Compitator agent 
builder = StateGraph(CompetitorState)
#Basically researchs about the compittator present in the industies of the topic 
# Compitator decision engine routes to various nodes like  web search , web scraping , draft reporting
builder.add_node('competitor_decision_engine' , competitor_decision_engine)

builder.add_edge(START , 'competitor_decision_engine')
builder.add_conditional_edges('competitor_decision_engine' ,
                              competitor_router , {
                                  "web_scraper": END ,
                                "web_search": END,
                                  "draft_report" : END
                              })



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