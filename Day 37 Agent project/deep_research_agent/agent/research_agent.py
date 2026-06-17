from graph.state import State
from tools.web_search_tools import web_search
from dotenv import load_dotenv
def research_agent(state : State) -> State:
    finding_content = []
    for tasks in state['plan']:
        print(tasks)
        task  = tasks['topic']
        priority =  tasks['priority']
        content = web_search.invoke(task)
        finding_content.append(
            {
                'topic' :  task ,
                'content' : content ,
                'priority' : priority
            }
        )
    return {
        'research_results' : finding_content
    }



if __name__ == '__main__':
    load_dotenv()
    response = research_agent({'plan' :
                    [{'topic': 'What is the definition of Tamil cinema and its historical significance in Indian film culture?',
   'priority': 'HIGH',
   'research_depth': 'HIGH'},
  {'topic': 'What are the core aesthetic, thematic, and technical concepts that characterize Tamil movies?',
   'priority': 'HIGH',
   'research_depth': 'HIGH'}]})
    print(response)