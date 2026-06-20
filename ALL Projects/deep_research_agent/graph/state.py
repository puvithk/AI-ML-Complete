from typing import TypedDict, Annotated
from pydantic import Field
from langgraph.graph import add_messages


class State(TypedDict):

    user_query : Annotated[str , Field(description="The query given by the user")]

    messages: Annotated[list[str],add_messages]

    draft_report: Annotated[str,Field(description="The current draft of the research report generated from the collected evidence.")]

    sub_questions: Annotated[list[str],Field(description="A list of research questions derived from the user's query to guide the research process.")]

    plan: Annotated[list[dict],Field(description="The research plan containing objectives, priorities, and tasks to be completed.")]

    research_results: Annotated[list[dict],Field(description="Raw information, findings, and data collected from various research sources.")]

    evidence: Annotated[list[dict],Field(description="Validated and organized evidence extracted from research results for report generation.")]

    need_more_research: Annotated[bool,Field(description="Indicates whether additional research is required based on the critic's evaluation.") ]

    iteration: Annotated[int,Field( 0, description="Tracks the number of research-review cycles completed.")]

    current_confidence : Annotated[float , Field(description="Confidance score of the current Draft report")]