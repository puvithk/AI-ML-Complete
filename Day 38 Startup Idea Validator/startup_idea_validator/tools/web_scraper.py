from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
tavily_search = TavilySearchResults(max_length =10)

@tool
def web_search_tool(query :str , seen_urls :list=[])-> list:
    """ Web search tools which provides the tool to search information in the internet 
    """
    #Result invoke by the tavily search api
    results =  tavily_search.invoke(query)
    # Seen url if called twice
    if seen_urls:
        # Check weather the result is repeated
        results = [
            r for r in results["results"]
            if r["url"] not in seen_urls
        ]
    # Return the result 
    return results

