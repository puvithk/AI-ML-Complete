from pydantic import BaseModel , Field
from graph.state import State
from utils.llm import llm
class DraftReport(BaseModel):

    draft_report : str =  Field(description="Complete Draft report of the user created by using the evidance")

    confidance_score : float = Field(description="Confidance of the draft report based on the evidnace 1-10 (1 means least and 10 means best)")




def draft_report(state : State)-> State:
    llm_structured_output = llm.with_structured_output(DraftReport)  


    response = llm_structured_output.invoke(
        f"""
       You are an expert Report Drafting Agent.

        Your responsibility is to create a detailed, professional, and
        well-structured research report using ONLY the evidence provided.

        User Query:
        {state['user_query']}

        Evidence:
        {state['evidence']}

        Instructions:
        - Use only the provided evidence.
        - Do not assume or invent information.
        - Do not add facts that are not present in the evidence.
        - Organize the report logically.
        - Use clear and professional language.
        - Include appropriate headings and subheadings.
        - Summarize findings accurately.
        - Ensure the report directly addresses the user's query.

        Report Structure:

        1. Introduction
        - Brief overview of the topic.
        - Explain the purpose of the report.

        2. Main Findings
        - Present all important findings.
        - Group related information together.
        - Use subsections where necessary.

        3. Analysis
        - Explain relationships between findings.
        - Highlight significant insights from the evidence.

        4. Conclusion
        - Summarize the key findings.
        - Provide a concise final answer to the user's query.

        Return only the final report in markdown format.
        Also proivde the confidance be frank
    """)

    return {
        "draft_report" :  response.model_dump()['draft_report'],
        "current_confidence" : response.model_dump()['confidance_score']
    }



if __name__ =="__main__":
    response =  draft_report({
        'evidence' :{'topic': 'Kannada Film Industry (Sandalwood)',
 'evidance_list': ['Kannada cinema, colloquially known as Sandalwood or Chandanavana, is based in Gandhi Nagar, Bengaluru, and produces motion pictures primarily in the Kannada language.',
  "The 1934 film 'Sati Sulochana', directed by Y. V. Rao, is recognized as the first Kannada talkie.",
  'The Kannada film industry experienced a significant milestone in 2022, reaching an 8-9 share of the Indian gross domestic box office, though this share declined in 2023.',
  "Historically, the 1970s and 1980s are widely considered the 'Golden Age' of Kannada cinema, marked by the rise of iconic stars like Dr. Rajkumar and Dr. Vishnuvardhan and the emergence of parallel cinema.",
  "The industry's thematic evolution has spanned from mythological and historical narratives to diverse genres including romance, action, and social issues, often drawing on local folklore and cultural heritage.",
  'Key structural challenges include a lack of consistent marketing, limited dubbing availability in the past, and an inconsistent distribution structure for smaller films.',
  "Recent commercial success has been driven by films like 'K.G.F' and 'Kantara', which gained traction by leveraging stories deeply rooted in Karnataka's unique local culture.",
  "The industry is currently navigating a period of recalibration, moving away from forced 'pan-India' branding toward prioritizing local audience connection and content-driven storytelling.",
  'Technological advancements, including digital distribution and the rise of OTT platforms, have provided new avenues for Kannada content to reach national and international audiences.']}
    ,
    'user_query' :  'Detailed explanation  about Kannada movie'})


    print(response)