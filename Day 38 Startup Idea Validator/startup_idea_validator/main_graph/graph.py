# Defination of the main graph 

#Import the required libaries 

from langgraph.graph import StateGraph , START , END 
from .state import MainState

from .nodes import pitch_summarizer


# Build a basic agent 
# Add the state 
builder = StateGraph(MainState)
# Create node of each agent or operations 
    # Question summarizer and decompose 

builder.add_node('pitch_summarizer' , pitch_summarizer)
    # Users approver 

    # Compitator agent 

    # Market agent 

    # Customer agent 

    # trend agent 


# Create egde and conditioanl edge 
builder.add_edge(START , 'pitch_summarizer')
builder.add_edge('pitch_summarizer' , END)
# Complile the build and add memory 

agent = builder.compile()


