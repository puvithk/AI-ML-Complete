from database_agent.agent.main_agent.graph import agent

config = {
    "configurable" :{
        "thread_id" : "1"
    }
}

from pprint import pprint

result = agent.invoke(
    {"user_question": "Give me top 5 sales"},
    config=config,
)

pprint(result, sort_dicts=False)
