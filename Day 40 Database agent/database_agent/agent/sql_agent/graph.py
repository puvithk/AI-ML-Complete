from langgraph.graph import StateGraph , START ,END
from agent.sql_agent.nodes import get_database_details , get_required_tables , get_table_details  , get_query , critic , execute_query , format_result
from agent.sql_agent.state import SqlQueryRetrialState
from agent.sql_agent.route import retry_query , is_query_successful
from utils.checkpointer import get_checkpointer
builder = StateGraph(SqlQueryRetrialState)
builder.add_node('database_details' , get_database_details)
builder.add_node('required_tables' , get_required_tables)
builder.add_node('table_details' , get_table_details)
builder.add_node('query_generator' , get_query)
builder.add_node('critic' , critic)
builder.add_node('excute_query' , execute_query)
builder.add_node('format_result' , format_result)

builder.add_edge(START , 'database_details')
builder.add_edge('database_details' , 'required_tables' )
builder.add_edge('required_tables' , 'table_details' )
builder.add_edge('table_details' , 'query_generator')
builder.add_edge('query_generator' , 'critic')
builder.add_edge('format_result' , END)
builder.add_conditional_edges('critic' ,
                              retry_query, 
                              {
                                 "excute_query" : "excute_query",
                                 "query_generator" : "query_generator",
                                 "format_result" : "format_result"
                              })

builder.add_conditional_edges('excute_query' , 
                              is_query_successful ,
                              {
                                  "query_generator" : "query_generator",
                                  "format_result" : "format_result"
                              }
                              )


agent = builder.compile(checkpointer=get_checkpointer())