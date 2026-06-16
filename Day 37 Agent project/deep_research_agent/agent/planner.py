
from pydantic import BaseModel , Field
from graph.state import State
from utils.llm import llm
class ResearchTask(BaseModel):
    topic : str = Field(description="Research topic or question")

    priority : str = Field(description="Priority of the topic : HIGH , MEDIUM , LOW")

    research_depth : str =Field(description="Depth at which the research should be done : HIGH , MEDIUM , LOW")


class ResearchPlan(BaseModel):
    tasks : list[ResearchTask]  = Field(description='List of research tasks to perform')




def research_planner(state : State) -> State:
    # Get the research plan done based on the topic and the priority and the requied depth of research 
    structured_output  =  llm.with_structured_output(ResearchPlan)


    reponse =  structured_output.invoke(
        f"""
        This is the list of question 
        {'\n'.join(state['sub_questions'])}
        
        Create a research plan.

            For each question:
            - Assign a priority (High/Medium/Low)
            - Assign research depth (High/Medium/Low)
            - Keep the original question as the topic

            Prioritize foundational questions first.

        """

        
    )


    return {
        "plan" : reponse.model_dump()['tasks']
    }

if __name__ == '__main__':
    print(research_planner(
        {"sub_questions" : [
    "What is LangGraph?",
    "What are the features of LangGraph?",
    "What are LangGraph limitations?",
    "How does LangGraph compare with CrewAI?"
]}
    ))