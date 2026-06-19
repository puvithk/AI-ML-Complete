from pydantic import BaseModel , Field

from utils.llm import llm

from graph.state import State



class Critic(BaseModel):
    need_more_research: bool = Field(
        description="Whether additional research is required."
    )

    feedback: str = Field(
        description="Explain what information is missing or why the report is sufficient."
    )

def critic(state: State)-> State:
    llm_structured_output = llm.with_structured_output(Critic)

    response = llm_structured_output.invoke(
        f"""
        You are a Research Report Critic Agent.

        Your responsibility is to evaluate whether the report
        adequately answers the user's query based on the available evidence.

        User Query:
        {state['user_query']}

        Draft Report:
        {state['draft_report']}

        Evaluate the report using the following criteria:

        - Does the report answer the user's query?
        - Are there important topics or subtopics missing?
        - Is the report detailed enough?
        - Are the conclusions supported by the evidence?
        - Would additional research significantly improve the report?

        Set need_more_research to:
        - True if important information is missing or the report is incomplete.
        - False if the report sufficiently answers the query.
        """
    )

    return {
            "need_more_research": response.need_more_research,
    "critic_feedback": response.feedback
    }




if __name__ =='__main__':
    print(critic({'draft_report' : """# Research Report: The Kannada Film Industry (Sandalwood)

## 1. Introduction
This report provides a comprehensive overview of the Kannada film industry, also known as Sandalwood or Chandanavana. Based in Bengaluru, Karnataka, this industry produces motion pictures primarily in the Kannada language. The purpose of this report is to summarize the industry's historical trajectory, cultural significance, and current challenges as presented in the provided evidence.

## 2. Main Findings

### Historical Evolution
The Kannada film industry has a rich history dating back to 1934 with the release of the first Kannada talkie film, 'Sati Sulochana'. Throughout its evolution, the industry has transitioned from early mythological and historical narratives to the significant parallel/art cinema movement of the 1970s and 80s. Recently, the industry has achieved international success through acclaimed films such as 'KGF', 'Kantara', and '777 Charlie'.

### Cultural Foundation
Key figures such as Dr. Rajkumar, Vishnuvardhan, and director Puttanna Kanagal are considered foundational to the industry's artistic development and cultural identity. Furthermore, the industry is deeply rooted in Kannada literature, frequently adapting novels and plays into motion pictures.

### Operational and Market Challenges
The industry faces several systemic hurdles:
* **Financial:** A heavy reliance on private film financiers due to limited institutional support, coupled with high investment failure rates and a lack of mid-budget film productions.
* **Market Competition:** Strong competition from industries like Tollywood and Kollywood, which often possess larger budgets and more extensive distribution networks.
* **Operational Constraints:** Inconsistent box office performance, a need for better theatrical support, and the ongoing requirement for content-driven, culturally rooted narratives.

### Digital Advancement
The emergence of OTT platforms and digital marketing has opened new channels for growth and distribution, though the industry still finds effective promotion challenging when compared to larger film markets.

## 3. Analysis
The Kannada film industry demonstrates a distinct tension between its deep cultural heritage and modern market demands. While the industry has successfully pivoted to international acclaim with recent content-driven blockbusters, it remains constrained by a polarized financial structure that lacks a stable mid-tier production model. The shift toward digital and OTT platforms offers a potential bridge for the industry to reach wider audiences; however, the reliance on traditional finance and the pressure from larger, more affluent neighboring industries remain significant obstacles to consistent growth.

## 4. Conclusion
The Kannada film industry is a historically significant and culturally rooted sector that has evolved from its mythological origins to achieving recent global recognition. Despite facing systemic financial challenges and intense competition from larger Indian film industries, the sector continues to leverage its literary heritage and emerging digital platforms to sustain its development. The future of the industry depends on its ability to foster consistent, content-driven narratives while overcoming institutional financial limitations.

    """
     , 
     'user_query' :' Detailed explanation  about Kannada movie' }))