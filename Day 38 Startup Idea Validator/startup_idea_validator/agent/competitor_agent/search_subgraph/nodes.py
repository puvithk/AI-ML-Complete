from .state import WebSearchedCleanedResult , WebSearchResultState
from utils.llm import llm

from .prompt import web_search_clean_prompt
from tools.web_tools import web_search_tool
def web_search_node(state : WebSearchResultState):
    
    response  = web_search_tool.invoke({
        'query' : state['question']
        })

    return {
        'raw_data' : response ,
  
    }
    
def clean_web_search(state : WebSearchResultState):

    
    llm_structure = llm.with_structured_output(WebSearchedCleanedResult)

    prompt =  web_search_clean_prompt.format(question=state['question'] , raw_data = state['raw_data'])


    respose = llm_structure.invoke(prompt)

    return {
        'sources' : respose.model_dump()['source'],
        'raw_data' : []
    }
