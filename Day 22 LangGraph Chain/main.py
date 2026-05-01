# %%
#Getting the gemeini generative model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage , HumanMessage, SystemMessage
from langgraph.graph import MessagesState , START ,END , StateGraph
from IPython.display import display , Image
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


llm_with_tools = llm.bind_tools([give_quote])


tool_result = llm_with_tools.invoke([HumanMessage(content="Give me todays best quote of a love life" , name="Puvith")])


# %%
class State(MessagesState):
    #Build in state 
    #Message State 
    # message : Annotated[list(messages) , add_message]
    # Here add_message will Append the message 
    pass

# %%
def tool_calling_llm(state : MessagesState):
    return {
        'messages' : llm_with_tools.invoke(state['messages'])
    }





agent = StateGraph(State)

agent.add_node("tool_calling_llm" , tool_calling_llm)


agent.add_edge(START , "tool_calling_llm")
agent.add_edge("tool_calling_llm" , END)




garph = agent.compile()





messages = garph.invoke({
    "messages" : HumanMessage("Hi how are you ")
})


# %%
messages = garph.invoke({
    "messages" : HumanMessage("Give me the best quote which says about the morning charm")
})




# %%



