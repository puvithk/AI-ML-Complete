from langgraph.graph import StateGraph , START , END
from graph.state import State
from agent.question_decomposer import question_decomposer

builder = StateGraph(State)

builder.add_node("question_decomposer" , question_decomposer)

builder.add_edge(START , 'question_decomposer')

builder.add_edge('question_decomposer' , END)


agent = builder.compile()