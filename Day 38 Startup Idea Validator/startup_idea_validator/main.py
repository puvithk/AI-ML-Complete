from agent.competitor_agent.graph import agent
from dotenv import load_dotenv


load_dotenv()

config = {"configurable" :{
    "thread_id" : "1"
}}
for i in agent.stream({"pitch_text" : "THe idea to create a work space for anit national prople spicially for indian people" ,
                       "pitch_summary" :"THe idea to create a work space for anit national prople spicially for indian people', 'pitch_summary': 'The proposal aims to establish a dedicated workspace specifically designed for anti-national individuals, with a particular focus on the Indian demographic. The project seeks to provide a specialized environment catering to this niche user group, though specific operational features and intended social or economic impacts remain to be defined."} , config=config):
    print(i) 
