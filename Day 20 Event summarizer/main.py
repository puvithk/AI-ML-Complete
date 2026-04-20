from llama_index.tools.google import GoogleCalendarToolSpec ,GmailToolSpec
from llama_index.core.workflow import Event, StartEvent, StopEvent, step, Workflow, Context
from llama_index.core.agent import ReActAgent  , AgentWorkflow
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.huggingface import HuggingFaceLLM
import llama_index.core
llama_index.core.set_global_handler("simple")
from datetime import datetime
import calendar
import asyncio
from dotenv import load_dotenv
load_dotenv()


# Google Calendar tools
google_tool = GoogleCalendarToolSpec(credentials_path='oauth.json' , 
                                    ).to_tool_list()
# Google gmail tools
gmail_tool = GmailToolSpec(credentials_path='oauth.json' , ).to_tool_list()
#Gemini LLM 
llm = GoogleGenAI(
        model='gemini-3-flash-preview'
)

#Creating calender agent 
calendar_agent = ReActAgent(
    tools= google_tool, #Google calender tools 
    name="CalenderEventFinder",# Name of the agent 
    description="Useful for finding, creating, or managing Google Calendar events.", #Description of the agent 
    llm=llm #LLm for desision making 
)
#Creating a Gmail agent 
gmail_agent = ReActAgent(
    tools=gmail_tool , #Google Gmail agent 
    name="GmailAgent", #Name of the agent 
    description="Useful for reading, searching, and sending emails via Gmail.",#Description of the agent 
    llm=llm
)
#If class was there itshere 


master_agent = AgentWorkflow(
    agents=[calendar_agent , gmail_agent],
    root_agent="CalenderEventFinder" , 

)

async def main():
    now = datetime.now()
    today_str = now.strftime("%B %d, %Y")
    country =  'INDIA UTC+5.30'
    last_day = calendar.monthrange(now.year, now.month)[1]
    end_of_month_str = now.replace(day=last_day).strftime("%B %d, %Y")

    query = f"""
            I need you to generate and email a comprehensive schedule summary.
            TODAY'S DATE: {today_str}
            TIMEZONE: {country}

            ### STEP 1: DATA COLLECTION
            1. Search my Google Calendar for all meetings and events scheduled for today ({today_str}).
            2. Search my Google Calendar for important non-meeting events from today until {end_of_month_str}. 
            Specifically look for: Deadlines, Exams, Submissions, and Milestones.

   

            ### STEP 2: DELIVERY
            Send an email with the following details:
            - TO: puvithkumar05@gmail.com
            - SUBJECT: TODAY'S EVENTS AND IMPORTANT DATES ({today_str})
            - BODY: The generated summary.

            Final confirmation: Confirm to me once the email has been successfully sent.
            """
    response = await master_agent.run(
        user_msg=query
    )
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())