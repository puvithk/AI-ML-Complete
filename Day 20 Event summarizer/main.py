from llama_index.tools.google import GoogleCalendarToolSpec ,GmailToolSpec
from llama_index.core.agent import ReActAgent  , AgentWorkflow
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import FunctionTool
import llama_index.core
import smtplib
from email.mime.text import MIMEText
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
llama_index.core.set_global_handler("simple")
from datetime import datetime
import calendar
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os 
# # Google gmail tools
# gmail_tool = GmailToolSpec(credentials_path='oauth.json' , ).to_tool_list()
# # Google Calendar tools
google_tool = GoogleCalendarToolSpec(credentials_path='oauth.json' , 
                                    ).to_tool_list()

#Gemini LLM 
llm = GoogleGenAI(
        model='gemini-3-flash-preview'
)

# llm = Anthropic(
#     model='claude-haiku-4-5-20251001',
#     api_key=os.environ.get("ANTHROPIC_API")
# )

# llm = OpenAI(
#     api_key=os.environ.get("OPENAI_API")
# )



# Define a Email sending Agent 
def send_mail(subject : str , body :str , to:str) -> str:
    """This function is used to send mail using Gmail
    
    Keyword arguments:
    subject -- Subject of the mail which need to be sent 
    body -- body string of the HTML which should be sent 
    to -- Email id of the person which we should send mail
    Return: return_description
    """
    smtp_server = "smtp.gmail.com"
    port = 587  # For STARTTLS
    sender_email = os.environ.get("SENDER_MAIL")
    password = os.environ.get("SENDER_PASSWORD")
    receiver_email = to

    # Create the message
    msg = MIMEText(body ,'html')
    msg["Subject"] =subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        # Connect and send
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")

email_sender = FunctionTool.from_defaults(
    send_mail , 
    name="MailSender",
    description="Usefull in sending messages thorugh gmail"
) 

#Creating calender agent 
calendar_agent = ReActAgent(
    tools= google_tool, #Google calender tools 
    name="CalenderEventFinder",# Name of the agent 
    description="Useful for finding, creating, or managing Google Calendar events.", #Description of the agent 
    llm=llm ,#LLm for desision making 
    verbose=True,  # ← See tool calls
    max_iterations=10
)
#Creating a Gmail agent 
gmail_agent = ReActAgent(
    tools=[email_sender] , #Google Gmail agent 
    name="GmailAgent", #Name of the agent 
    description="Useful for sending emails via Gmail.",#Description of the agent 
    llm=llm,
    verbose=True,  # ← See tool calls
    max_iterations=10
)
#If class was there itshere 
master_agent = AgentWorkflow(
    agents=[calendar_agent , gmail_agent],
    root_agent="CalenderEventFinder" , 
    verbose=True 

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
            You are a scheduling assistant. 
            CRITICAL RULE: Never guess or hallucinate calendar events. 
            You must ALWAYS call the 'list_events' or 'load_data' tool from the CalenderEventFinder 
            to get real data before moving to the GmailAgent.
            ### STEP 1: DATA COLLECTION
            1. Search my Google Calendar for all meetings and events scheduled for today ({today_str}).
            2. Search my Google Calendar for important non-meeting events from today until {end_of_month_str} (Deadlines, Exams, Submissions, and Milestones).

            ### STEP 2: EMAIL PREPARATION
            Use the following HTML template for the email body. Replace the placeholders with the data you retrieved:

            ```html
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                .container {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
                .header {{ background-color: #1a73e8; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; color: #3c4043; }}
                .section-title {{ color: #1a73e8; border-bottom: 2px solid #f1f3f4; padding-bottom: 8px; margin-top: 25px; font-size: 18px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th {{ text-align: left; background-color: #f8f9fa; color: #5f6368; padding: 10px; font-size: 14px; border-bottom: 1px solid #dee2e6; }}
                td {{ padding: 12px 10px; border-bottom: 1px solid #f1f3f4; font-size: 14px; }}
                .footer {{ background-color: #f8f9fa; color: #70757a; font-size: 12px; text-align: center; padding: 15px; }}
            </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2 style="margin:0;">Schedule Summary</h2>
                        <p style="margin:5px 0 0 0; opacity: 0.9;">{today_str}</p>
                    </div>
                    <div class="content">
                        <div class="section-title">📅 Today's Meetings</div>
                        <table>
                            <thead><tr><th>Time</th><th>Event</th><th>Location</th></tr></thead>
                            <tbody>[INSERT_TODAY_ROWS]</tbody>
                        </table>

                        <div class="section-title">🚀 Upcoming Milestones & Deadlines</div>
                        <p style="font-size: 12px; color: #70757a;">Until {end_of_month_str}</p>
                        <table>
                            <thead><tr><th>Date</th><th>Task / Milestone</th></tr></thead>
                            <tbody>[INSERT_UPCOMING_ROWS]</tbody>
                        </table>
                    </div>
                    <div class="footer">
                        Automated Assistant • {country} Timezone
                    </div>
                </div>
            </body>
            </html>
            STEP 3: DELIVERY
            Send the email with these details:

            TO: puvithkumar05@gmail.com

            SUBJECT: TODAY'S EVENTS AND IMPORTANT DATES ({today_str})

            BODY: The completed HTML from Step 2.

            Final confirmation: Confirm once the email has been successfully sent.
            """
    response = await master_agent.run(
        user_msg=query 
    )
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())