from typing import TypedDict , Annotated
from pydantic import Field  , BaseModel



class CompetiatorState:
    #The idea query 
    pitch_query :Annotated[str , Field(description="User givedn pitch query")]
    # Picth summary #Given by the previous agent 
    pitch_summary : Annotated[str , Field(description="The summary of the full pitch query")]
    #Question which should be asked 

    questions : Annotated[list[str] , Field(description="Question which should be addressed")]

    # Detailed evidance 
    evidance :Annotated[list[dict] , Field(description="Evidance given by the agent for carring out task")]
    #Source of the summary and evidance 
    sources :Annotated[list[dict] , Field(description="Source collected by various resourses")]

    #Detailed summary of the research

    report :Annotated[str , Field(description="Final Summary report of the current agent")] 




