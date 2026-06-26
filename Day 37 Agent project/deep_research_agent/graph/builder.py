from langgraph.graph import StateGraph , START , END
from graph.state import State
from agent.question_decomposer import question_decomposer
from agent.planner import research_planner
from agent.research_agent import research_agent
from agent.evidence_collector import evidence_collector
from agent.draft_report import draft_report
from agent.critic  import critic
from nodes.router import should_continue_research
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
builder = StateGraph(State)
builder.add_node("question_decomposer" , question_decomposer)
builder.add_node("planner" , research_planner)
builder.add_node("research_agent", research_agent)
builder.add_node("evidence_collector" , evidence_collector)
builder.add_node("draft_report" , draft_report)
builder.add_node('critic' , critic)
builder.add_edge(START , 'question_decomposer')
builder.add_edge('question_decomposer' , 'planner')
builder.add_edge('planner' , 'research_agent')
builder.add_edge('research_agent' , 'evidence_collector')
builder.add_edge('evidence_collector' ,"draft_report")
builder.add_edge('draft_report' , 'critic')
builder.add_conditional_edges(
    'critic',
    should_continue_research , 
    {
        "research_agent": "research_agent",  
        "__end__": END
    })
agent = builder.compile(checkpointer=memory)