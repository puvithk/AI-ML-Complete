from llama_index.core  import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter



pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0) , 
        HuggingFaceEmbedding(
              model_name="BAAI/bge-small-en-v1.5"
        ),
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



async def main():
    model = await pipeline.arun(documents = source_docs)
    return model

# Run the async function
import asyncio
model = asyncio.run(main())