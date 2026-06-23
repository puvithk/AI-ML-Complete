#Defining the main state in the graph 
from typing  import TypedDict , Annotated
from pydantic import Field
class MainState(TypedDict):
    #Plain user input 

    pitch_text : Annotated[str , Field(description="Plain pitch text given as input by the user")]

    # Summary of the pitch text 

    pitch_summary : Annotated[str , Field(description="Detailed explanation of the pitch or the idea which the research should be carried out")]


    # Approved buy the user 


    user_approved : Annotated[bool , Field(description="Weather the final summary if approved by the user or not")]


    # Other agents output 

    