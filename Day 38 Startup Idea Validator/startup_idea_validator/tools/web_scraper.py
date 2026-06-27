from langchain_community.tools import TavilySearchResults 
from langchain_tavily.tavily_crawl import TavilyCrawl
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

tavily_search = TavilySearchResults(max_length =10)

tavily_crawl = TavilyCrawl(max_depth=1, max_breadth=10, limit=10)

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


@tool
def web_scraping_tool(instructions : str , url:str = None):
    """Web scraping tool used to get the details from the webpage based on the instruction and the urls
    """

    tavily_crawl.instructions =  instructions


    response = tavily_crawl.invoke({
        'url' : url
    })

    if response:
        response['result'] = [ {"url" : i['url'] , "raw_content" : i['raw_content']} for i in response['results']]

    return response


if __name__ =='__main__':
    response = web_scraping_tool.invoke({
    "instructions": "Find merits of langchain",
    "url": "https://docs.langchain.com/oss/python/build-overview"
})
    print(response)
    print(type(response))