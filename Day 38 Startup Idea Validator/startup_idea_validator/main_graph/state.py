#Defining the main state in the graph 
from typing  import TypedDict , Annotated
from pydantic import Field , BaseModel
class MainState(TypedDict):
    #Plain user input 

    pitch_text : Annotated[str , Field(description="Plain pitch text given as input by the user")]

    # Summary of the pitch text 

    pitch_summary : Annotated[str , Field(description="Detailed explanation of the pitch or the idea which the research should be carried out")]

    pitch_summary_feedback : Annotated[str , Field(description="The feedback given  by the user for he current summary ")]
    # Approved buy the user 


    user_approved_summary : Annotated[bool , Field(description="Weather the final pitch summary is approved by the user or not")]


    # Other agents output 

    
class PitchSummary(BaseModel):

    pitch_summary : str = Field(description="Pitch summary in depth info about the current idea")


