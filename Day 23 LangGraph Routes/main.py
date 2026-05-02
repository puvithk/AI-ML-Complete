
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage , HumanMessage, SystemMessage
from langgraph.graph import MessagesState , START ,END , StateGraph
from IPython.display import display , Image
from langgraph.prebuilt import ToolNode , tools_condition
from pprint import pprint 
from dotenv import load_dotenv
load_dotenv()




llm = ChatGoogleGenerativeAI(
    model = 'gemini-3-flash-preview'
)



def give_quote(topic : str):
    """Function used to give a quote based on the topic given 
    
    Keyword arguments:
    topic -- str : topic of which the quote must be given 
    Return: the quote based ojn the topic 
    """
    return "Be happy always a day without laugter is the day wasted "



class State(MessagesState):
    pass


llm_with_tools= llm.bind_tools([give_quote])


# %%
def tool_calling_llm(state : State):
    return {
        'messages' : llm_with_tools.invoke(state['messages'])
    }


# %%
builder = StateGraph(State)

# %%

builder.add_node('tool_calling_llm' , tool_calling_llm)
builder.add_edge(START , 'tool_calling_llm')
builder.add_node('tools' , ToolNode([give_quote]))
builder.add_conditional_edges('tool_calling_llm' ,  tools_condition)

builder.add_edge('tools' , END)
graph = builder.compile()


# %%
result = graph.invoke({
    'messages':HumanMessage("Wht is moring quote" , name="puvith")
})





