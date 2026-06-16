from langgraph.graph import StateGraph , START , END
from graph.state import State
from agent.question_decomposer import question_decomposer
from agent.planner import research_planner
builder = StateGraph(State)

builder.add_node("question_decomposer" , question_decomposer)
builder.add_node("planner" , research_planner)
builder.add_edge(START , 'question_decomposer')
builder.add_edge('question_decomposer' , 'planner')
builder.add_edge('planner' , END)


agent = builder.compile()