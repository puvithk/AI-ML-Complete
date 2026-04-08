from smolagents import InferenceClientModel , ToolCallingAgent ,CodeAgent ,  FinalAnswerTool , WebSearchTool  , DuckDuckGoSearchTool  
agent =  CodeAgent(tools=[FinalAnswerTool() , DuckDuckGoSearchTool()] , model=InferenceClientModel())
agent.run("Wht is the climate today in mangalore ")