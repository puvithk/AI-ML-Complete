from typing import TypedDict , Annotated , Literal
from pydantic import Field  , BaseModel


import operator
class CompetitorState(TypedDict):
    #The idea query 
    pitch_query :str 
    # Picth summary #Given by the previous agent 
    pitch_summary : str 
    #Question which should be asked 

    questions : Annotated[list[str] ,operator.add]

    # Detailed evidance 
    evidance :Annotated[list[dict]  ,operator.add]
    #Source of the summary and evidance 
    sources :Annotated[list[dict] ,operator.add ]

    #Detailed summary of the research

    report :str 

    decision_feedback : str 
    decision: Literal["web_scraper", "web_search", "draft_report"] 


    raw_data: Annotated[list[dict], operator.add]



class CompetitorDecision(BaseModel):
    decision: Literal["web_scraper", "web_search", "draft_report"] = Field(description="Field which decides which function or tool should run")

    decision_feedback : str = Field(description="What should be done based on the current desision")


class QuestionDecomposed(BaseModel):
    # Question for the web search
    questions : list[str] =  Field(description="All the questions based on the user request")


