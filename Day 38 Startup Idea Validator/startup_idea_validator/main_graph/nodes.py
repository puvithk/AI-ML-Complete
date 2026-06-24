# Creating various nodes for the graph
from utils.llm import llm
from .state import MainState , PitchSummary
def pitch_summarizer(state : MainState)-> MainState:
    previous_summary = state.get("pitch_summary", "")
    previous_feedback = state.get("pitch_summary_feedback", "")
    prompt = f"""Your are a pitch summaizring agent 
    Your are responsible for creating a complete summary of the pitch by the given pitch text
    This is the pitch provided by the user : {state['pitch_text']}


    Provide the overall summary and breif explanation based on the query
    Note :
    -Only summarize the pitch desk dont add other things 
    -A concise, objective overview of the idea, product, or proposal, 
    highlighting the problem being addressed, the solution, 
    key features, target users, and expected impact. Avoid phrases such as "you're saying,"
    "you imply," or any interpretation of the speaker's intent. 
    Present only the core points of the pitch in a neutral summary.

    This is the previous summary and user feedback (User this if its provided only)
    previoues Pitch summary : {previous_summary}

    feedback for previous summary : {previous_feedback}

    """

    llm_structure = llm.with_structured_output(PitchSummary)
    
    response = llm_structure.invoke(prompt)

    return {
        'pitch_summary' : response.model_dump()['pitch_summary']
    }


