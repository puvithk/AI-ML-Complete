from langchain_community.docstore.document import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent , DuckDuckGoSearchTool , OpenAIServerModel, InferenceClientModel , FinalAnswerTool , LiteLLMModel


class ProjectDetailsRetriverTool(Tool):
    name  =  'personal_propject_retrival'
    description = """
    Use this tool to answer ANY question about personal projects.
    You MUST call this tool before answering.
    Do NOT answer from your own knowledge.
    """    
    inputs = {
        "query" :{
            "type" :"string",
            "description" : "The query to perform. This should be a query related to my personal project details"
        }
    }
    output_type = 'string'


    def __init__(self , docs , **kwargs):
        super().__init__(**kwargs)
        self.retirver = BM25Retriever.from_documents(
            docs , k = 4
        )
    
    def forward(self , query:str)->str:
        assert isinstance(query , str) ,  "Your serach query must be string"


        docs = self.retirver.invoke(
            query
        )
        return "\nRetieved Ideas :\n" + "".join(
            [
                f'\n\n==== Idea {str(i)}====\n' + doc.page_content
                for i , doc in enumerate(docs)
            ]

        )


projectsIdeas  = [
    {
        "title": "AyurVana",
        "domain": "AI + Web Development + Healthcare",
        "text": "AyurVana is a digital platform designed to educate users about medicinal plants used in AYUSH systems. It includes features like plant information pages, a virtual garden, AI chatbot, bookmarking system, and a lens feature for plant identification."
    },
    {
        "title": "Blockchain-Based Evidence Tracker",
        "domain": "Blockchain + Web Development",
        "text": "A secure system for tracking and managing evidence using Hyperledger Fabric. It ensures tamper-proof storage and transparency. Built with Node.js backend and React frontend."
    },
    {
        "title": "Placement Preparation Planner",
        "domain": "Web Development",
        "text": "A planning tool that helps students organize and track their placement preparation. It includes features for managing study schedules, tracking progress, and organizing placement-related activities."
    },
    {
        "title": "Enhanced Diagnostic Tool for Youth Depression Recovery",
        "domain": "AI/ML + Healthcare",
        "text": "An AI-powered system for early detection of youth depression using CNN for facial emotion recognition, Naive Bayes for text analysis, and Random Forest for mood detection. It also integrates wearable data for personalized insights."
    },
    {
        "title": "SmartClass Attend",
        "domain": "AI/ML + Computer Vision",
        "text": "An automated attendance system using face recognition (FaceNet + KNN). Built with Flask and React, it enables real-time attendance tracking and classroom monitoring."
    },
    {
        "title": "AI-Powered Solution for the Visually Impaired",
        "domain": "AI + IoT + Assistive Technology",
        "text": "A smart system using AI, sensors, and IoT to assist visually impaired individuals. It provides obstacle detection, face recognition, printed text reading, and multilingual voice alerts through smart glasses and earbuds."
    },
    {
        "title": "SareeFusion",
        "domain": "AI + Design Automation",
        "text": "A saree design automation tool that allows users to combine different parts of sarees (pallu, border, body) and generate weaving graphs for handloom production, enabling personalized design creation."
    }
]


source_docs = [
    Document(page_content=project['text'] ,metadata = {'title' : project['title'] ,'domain' : project['domain']} )
    for project in projectsIdeas
]

text_spiltter =  RecursiveCharacterTextSplitter(
     chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs_processed = text_spiltter.split_documents(source_docs)
personal_propject_retrival = ProjectDetailsRetriverTool(docs_processed)

# model = LiteLLMModel(
#     model_id="ollama_chat/qwen2:7b", 
#     api_base="http://127.0.0.1:11434",  
#     num_ctx=8192,
# )
model = OpenAIServerModel(model_id='gpt-4o')
agent = CodeAgent(tools=[personal_propject_retrival  , FinalAnswerTool() , DuckDuckGoSearchTool()] , model=model)


agent.run("Is there any project with AI ML topics in my , List them ")

