from langgraph.graph import StateGraph , START , END
from graph.state import State
from agent.question_decomposer import question_decomposer
from agent.planner import research_planner
from agent.research_agent import research_agent
from agent.evidence_collector import evidence_collector
builder = StateGraph(State)

builder.add_node("question_decomposer" , question_decomposer)
builder.add_node("planner" , research_planner)
builder.add_node("research_agent", research_agent)
builder.add_node("evidence_collector" , evidence_collector)
builder.add_edge(START , 'question_decomposer')
builder.add_edge('question_decomposer' , 'planner')
builder.add_edge('planner' , 'research_agent')
builder.add_edge('research_agent' , 'evidence_collector')
builder.add_edge('evidence_collector' ,END)

agent = builder.compile()