
import os 
import smtplib
from email.mime.text import MIMEText
# Define a Email sending Agent 
from dotenv import load_dotenv
load_dotenv()
def send_mail(subject : str , body :str , to:str) -> str:
    """This function is used to send mail using Gmail
    
    Keyword arguments:
    subject -- Subject of the mail which need to be sent 
    body -- body string of the mail which should be sent 
    id -- Email id of the person which we should send mail
    Return: return_description
    """
    smtp_server = "smtp.gmail.com"
    port = 587  # For STARTTLS
    sender_email = os.environ.get("SENDER_MAIL")
    password = os.environ.get("SENDER_PASSWORD")
    receiver_email = to

    # Create the message
    msg = MIMEText(body)
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



send_mail("Testing mail" , "hello guys welcome to channel " , 'puvithkumar05@gmail.com')