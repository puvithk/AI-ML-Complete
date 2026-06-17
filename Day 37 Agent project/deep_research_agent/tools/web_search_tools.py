from langchain_community.tools import TavilySearchResults
from graph.state import State
from langchain.tools import tool
from dotenv import load_dotenv


@tool
def web_search(query: str) -> str:
    """
    Search the web for information related to a query.
    """
    tavily_search_result = TavilySearchResults()

    results = tavily_search_result.invoke(query)
    

    return results

    

if __name__ =="__main__":
    load_dotenv()
    print((web_search("Puvith kumar the ai engineer")))