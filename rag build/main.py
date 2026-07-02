import uuid
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import process_pdf
from vector_store import STORE
from agent import run_query

app = FastAPI(title="Section-Parallel RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    pdf_bytes = await file.read()
    doc_id = str(uuid.uuid4())[:8]

    t0 = time.time()
    sections = process_pdf(pdf_bytes, doc_id)
    if not sections:
        raise HTTPException(422, "Could not extract any sections from this PDF")
    STORE.add_document(doc_id, sections)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_sections": len(sections),
        "num_chunks": sum(len(s.chunks) for s in sections),
        "processing_ms": elapsed_ms,
        "sections": [
            {"section_id": s.section_id, "title": s.title, "priority": s.priority, "num_chunks": len(s.chunks)}
            for s in sections
        ],
    }


class QueryRequest(BaseModel):
    doc_id: str
    query: str
    api_key: str = ""


@app.post("/query")
async def query(req: QueryRequest):
    if not STORE.has_document(req.doc_id):
        raise HTTPException(404, "Unknown doc_id -- upload the PDF first")
    result = await run_query(req.doc_id, req.query, req.api_key)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
