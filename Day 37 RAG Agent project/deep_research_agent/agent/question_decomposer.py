from utils.llm import llm
from typing import Annotated ,TypedDict
from pydantic import Field , BaseModel
from graph.state import State

class QuestionDecomposer(BaseModel):
    questions: list[str] = Field(
        description="""
        Generate a list of specific research questions from the user's query.
        The questions should collectively cover all key topics and subtopics
        required to provide a complete and well-researched answer.
        """
    )

def question_decomposer(state : State) -> State:
    #Get the sub question with sub questions
    structured_output  = llm.with_structured_output(QuestionDecomposer)

    result = structured_output.invoke(f"""
        User query 
                                      {state['user_query']}    
        Break this query into multiple research questions.    
        """)
    
    return {
        "sub_questions" : result.model_dump()["questions"]
    }