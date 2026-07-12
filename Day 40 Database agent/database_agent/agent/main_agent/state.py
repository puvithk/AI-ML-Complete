from typing import TypedDict ,Annotated , Literal
import operator
from pydantic import BaseModel , Field

class OrchestratorAgentState(TypedDict):
    user_question : str

    decomposed_question : list[str] #Optional if there is a complex question need to decide it 


    final_answer : Annotated[list[str] , operator.add] # Output from each of the agent 


    summarized_output : str # Formated summarized output 


    summary_verified : bool

    revision_count : int  # how many critic->decompose cycles have run (loop guard)

    next_route : str 

    memeory : str

    working_summary : str

class RewriteAndRoute(BaseModel):
    next_route : str = Field(description="Which is the next route the agent must folow")
    rewrittern_query : str = Field(description="The rewriten query base don the user input")
    

class DecomposerResult(BaseModel):

    questions : list[str] = Field(description="All the sub question for the sql agent") 

class FinalFormatedResult(BaseModel):

    final_answer : str = Field(description="Final Answer Given by combining all the result")


class CriticRouteState(BaseModel):

    next_node : Literal['final_result' , 'question_decomposer' ] =Field(description="Which node should the agent must follow")


class ChitChatResponse(BaseModel):
    answer : str =Field(description="Result of the response for normal text")