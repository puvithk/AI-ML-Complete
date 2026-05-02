
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode , tools_condition
from langchain.messages import HumanMessage
from langgraph.graph import START , StateGraph , END , MessagesState 
from IPython.display import display , Image
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite-preview'
)



# %%
def multiply(a:int , b:int) -> int:
    """Multiply a and b 
    
    Keyword arguments:
    a -- number 1
    b -- number 2 
    Return: result after multipling 
    """
    return a*b

def add(a:int , b:int) -> int :
    """Add a and b 
    
    Keyword arguments:
    a -- number 1
    b -- number 2
    Return: result after adding 
    """
    return a+b 

    



# %% [markdown]
# Bind the tools with the llm 

# %%
llm_with_tools = llm.bind_tools([multiply , add])

# %% [markdown]
# State for the graph 

# %%
class State(MessagesState):
    pass

# %% [markdown]
# Tool calling Llm  function
# 

# %%
def tools_calling_llm(state : State):
    return {
        'messages' : llm_with_tools.invoke(state['messages'])
    }

# %% [markdown]
# Build the agent 
# 

# %%
builder = StateGraph(State)
builder.add_node("tools_calling_llm" , tools_calling_llm)

builder.add_edge(START , "tools_calling_llm")
builder.add_node('tools' , ToolNode([multiply , add]))
builder.add_conditional_edges(
    "tools_calling_llm" , 
    tools_condition
)
builder.add_edge('tools' , "tools_calling_llm")
builder.add_edge("tools_calling_llm" , END)

# %%
agent = builder.compile()

# %%
agent

# %%
display(Image(agent.get_graph().draw_mermaid_png()))

# %%
result = agent.invoke(
    {
        "messages" : HumanMessage("What is 10 add by 2 multiplyed by 2" , name="puvith")
    }
)





