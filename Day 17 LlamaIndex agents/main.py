from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow  import  AgentWorkflow 
from llama_index.core.workflow import Context
import dotenv
dotenv.load_dotenv()
from llama_index.core.tools import FunctionTool

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import QueryEngineTool
import chromadb
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()


def build_index():
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


    # LLM setup
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ['GOOGLE_API_KEY']
    )
    collection_found = False 
    # Chroma DB setup
    chroma = chromadb.PersistentClient(path='./personal_db')
    if(chroma.get_collection('puvith')):
        chroma_collection = chroma.get_collection('puvith')
        collection_found = True
    else :
        chroma_collection = chroma.get_or_create_collection('puvith')
        collection_found = False


    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # Embedding model
    embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter( chunk_overlap=0),
            embedding_model,
        ],
        vector_store=vector_store
    )

    # Convert to documents
    source_docs = [
        Document(
            text=project['text'],
            metadata={
                'title': project['title'],
                'domain': project['domain']
            }
        )
        for project in projectsIdeas
    ]

    # Run pipeline (async inside sync function)
    async def run_pipeline():
        await pipeline.arun(documents=source_docs)
    if not collection_found:
        asyncio.run(run_pipeline())

    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding_model
    )

    return index


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"


tool = FunctionTool.from_defaults(
    get_weather , 
    name='get_weather',
    description="Useful for getting the weather for a given location."
)

llm = GoogleGenAI(
    model="gemini-3-flash-preview"
)


index = build_index()
queryEngine = index.as_query_engine(llm=llm)
queryTool = QueryEngineTool.from_defaults(queryEngine , name="presonal_project" , description="My peronsal project which i have done in my life time ")
agent = AgentWorkflow.from_tools_or_functions(
[tool , queryTool],
    llm=llm)


ctx = Context(agent)
context = dict
async def main():
    global context
    # response = await agent.run("My name is Puvith" , ctx = ctx)

    # print(response)
    # response = await agent.run("Whats my name" , ctx=ctx)
    # print(response)

    response = await agent.run("What are my presonal project which is related to web devoploment ?")
    print(response)
 

import asyncio
asyncio.run(main())
