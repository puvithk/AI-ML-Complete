# Defination of the main graph 

#Import the required libaries 

from langgraph.graph import StateGraph , START , END 
from .state import MainState
from langgraph.checkpoint.memory import InMemorySaver
from .nodes import pitch_summarizer , check_user_approval , human_approval

# Build a basic agent 
# Add the state 
builder = StateGraph(MainState)
# Create node of each agent or operations 
    # Question summarizer and decompose 

builder.add_node('pitch_summarizer' , pitch_summarizer)
    # Users approver 
builder.add_node('human_approval' , human_approval )
    # Compitator agent 

    # Market agent 

    # Customer agent 

    # trend agent 


# Create egde and conditioanl edge 
builder.add_edge(START , 'pitch_summarizer')
builder.add_edge('pitch_summarizer' , 'human_approval')
builder.add_conditional_edges('human_approval' , 
                                check_user_approval,
                            {
                                'END' : END ,
                                'pitch_summarizer' : 'pitch_summarizer'
                            }
                                                        )



# Complile the build and add memory 
memory = InMemorySaver()
agent = builder.compile( checkpointer=memory)


