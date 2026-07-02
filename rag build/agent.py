"""
LangGraph orchestration -- Gemini-backed, with an LLM relevance gate.

Graph shape:

    START --Send(one per section)--> retrieve_and_grade (fan-out, parallel)
         retrieve_and_grade --> synthesize (fan-in / reduce)
    synthesize --> END

`Send` is LangGraph's map-reduce primitive: it spins up one execution of
`retrieve_and_grade` per section *concurrently*, each with its own isolated
state slice -- that's the "each section must run in parallel" requirement.

Each section's node does two things, both scoped to that section only:
  1. Local TF-IDF retrieval (fast, no network).
  2. An LLM "relevance gate": Gemini is shown ONLY that section's query +
     retrieved chunks and asked to reason about whether they actually
     answer the query, returning {relevant, reasoning, confidence}. A
     section only advances into the final answer if the LLM says it's
     relevant -- a high TF-IDF score alone is not enough. This filters out
     sections that are lexically similar but not actually on-topic.

The synthesize step then only uses sections that passed the gate, ranked
by relevance x priority, and asks Gemini to write the final cited answer.
"""
import os
import re
import json
import time
from typing import Annotated, List, TypedDict, Optional
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from vector_store import STORE, RetrievedChunk
from pdf_processor import Section

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


TOP_K_PER_SECTION = 3
MAX_SECTIONS_TO_LLM = 6  # cap for the final synthesis prompt size / latency
GEMINI_MODEL = "gemini-2.5-flash"  # fast/cheap, good for high-volume parallel grading calls


class SectionResult(TypedDict):
    section_id: str
    title: str
    priority: float
    chunks: List[RetrievedChunk]
    weighted_score: float
    relevant: bool
    reasoning: str
    confidence: float


class GraphState(TypedDict):
    doc_id: str
    query: str
    api_key: str
    section_results: Annotated[List[SectionResult], operator.add]
    answer: str
    timing_ms: dict


def _client(api_key: str):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or genai is None:
        return None
    return genai.Client(api_key=key)


def _fan_out(state: GraphState):
    sections = STORE.get_sections(state["doc_id"])
    return [
        Send("retrieve_and_grade", {
            "doc_id": state["doc_id"],
            "query": state["query"],
            "api_key": state.get("api_key", ""),
            "section": s,
        })
        for s in sections
    ]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_relevance_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_RE.search(text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _grade_relevance(client, query: str, section_title: str, chunks: List[RetrievedChunk]) -> dict:
    """Ask Gemini to reason about whether this section's retrieved chunks
    are actually relevant to the query, scoped to this section only. Runs
    once per section, concurrently with every other section's grading call."""
    excerpt = "\n".join(f"- {c.text}" for c in chunks)
    prompt = (
        "You are a relevance judge for a retrieval system. You are shown ONE section "
        "from a larger document and the excerpts pulled from it for a user's query.\n\n"
        f"USER QUERY: {query}\n\n"
        f"SECTION TITLE: {section_title}\n"
        f"RETRIEVED EXCERPTS:\n{excerpt}\n\n"
        "Reason briefly about whether these excerpts genuinely help answer the query "
        "(not just share vocabulary with it). Then respond with ONLY a JSON object, "
        "no markdown fences, in exactly this shape:\n"
        '{"relevant": true or false, "confidence": 0.0-1.0, "reasoning": "one short sentence"}'
    )
    try:
        resp = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=200,
                temperature=0.0,
            ),
        )
        parsed = _parse_relevance_json(resp.text)
        if parsed and "relevant" in parsed:
            return {
                "relevant": bool(parsed["relevant"]),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
    except Exception as e:
        return {"relevant": True, "confidence": 0.0, "reasoning": f"LLM grading failed ({e}); kept by default"}
    return {"relevant": True, "confidence": 0.0, "reasoning": "Could not parse grading response; kept by default"}


async def retrieve_and_grade(payload: dict) -> GraphState:
    doc_id, query, section = payload["doc_id"], payload["query"], payload["section"]
    api_key = payload.get("api_key", "")
    section: Section

    idx = STORE.get_section_index(doc_id, section.section_id)
    chunks = idx.search(query, top_k=TOP_K_PER_SECTION)
    top_sim = max((c.score for c in chunks), default=0.0)
    weighted = top_sim * (0.5 + 0.5 * section.priority)

    client = _client(api_key)
    if chunks and client is not None:
        grade = await _grade_relevance(client, query, section.title, chunks)
    elif chunks:
        grade = {
            "relevant": top_sim >= 0.15,
            "confidence": top_sim,
            "reasoning": "No Gemini API key provided; used similarity threshold instead of LLM reasoning.",
        }
    else:
        grade = {"relevant": False, "confidence": 0.0, "reasoning": "No matching chunks in this section."}

    result: SectionResult = {
        "section_id": section.section_id,
        "title": section.title,
        "priority": section.priority,
        "chunks": chunks,
        "weighted_score": weighted,
        "relevant": grade["relevant"],
        "reasoning": grade["reasoning"],
        "confidence": grade["confidence"],
    }
    return {"section_results": [result]}


async def synthesize(state: GraphState) -> GraphState:
    t0 = time.time()
    all_results = state["section_results"]
    passed = [r for r in all_results if r["chunks"] and r["relevant"]]
    passed.sort(key=lambda r: (r["weighted_score"] * (0.5 + 0.5 * r["confidence"])), reverse=True)
    top_results = passed[:MAX_SECTIONS_TO_LLM]

    context_blocks = []
    for r in top_results:
        joined = "\n".join(f"- {c.text}" for c in r["chunks"])
        context_blocks.append(f"[Section: {r['title']} | priority={r['priority']:.2f}]\n{joined}")
    context = "\n\n".join(context_blocks) if context_blocks else "No section passed the relevance gate."

    api_key = state.get("api_key", "")
    client = _client(api_key)
    answer = ""
    if client is not None and context_blocks:
        prompt = (
            "Answer the user's question using ONLY the section excerpts below -- these have "
            "already passed a relevance check, and are ranked by relevance x priority. Prefer "
            "higher-ranked sections when they conflict. Cite the section title(s) you used in "
            "brackets.\n\n"
            f"QUESTION: {state['query']}\n\nSECTION EXCERPTS:\n{context}\n\nANSWER:"
        )
        resp = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(max_output_tokens=700, temperature=0.2),
        )
        answer = resp.text or ""
    elif not context_blocks:
        answer = "No section passed the relevance check for this query -- nothing on-topic was found."
    else:
        top = top_results[0]
        answer = (
            f"(No Gemini API key provided -- showing top extractive match that passed the "
            f"similarity threshold.)\n\nFrom \"{top['title']}\": {top['chunks'][0].text}"
        )

    return {
        "answer": answer,
        "timing_ms": {**state.get("timing_ms", {}), "synthesize": round((time.time() - t0) * 1000, 1)},
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve_and_grade", retrieve_and_grade)
    graph.add_node("synthesize", synthesize)
    graph.add_conditional_edges(START, _fan_out, ["retrieve_and_grade"])
    graph.add_edge("retrieve_and_grade", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


AGENT = build_graph()


async def run_query(doc_id: str, query: str, api_key: str = ""):
    t0 = time.time()
    result = await AGENT.ainvoke({
        "doc_id": doc_id,
        "query": query,
        "api_key": api_key,
        "section_results": [],
        "answer": "",
        "timing_ms": {},
    })
    total_ms = round((time.time() - t0) * 1000, 1)
    # Relevant (gate-passed) sections first, then filtered-out ones, each ranked within their group.
    ranked = sorted(
        result["section_results"],
        key=lambda r: (r["relevant"], r["weighted_score"] * (0.5 + 0.5 * r["confidence"])),
        reverse=True,
    )
    return {
        "answer": result["answer"],
        "sections": [
            {
                "section_id": r["section_id"],
                "title": r["title"],
                "priority": r["priority"],
                "weighted_score": round(r["weighted_score"], 4),
                "relevant": r["relevant"],
                "confidence": round(r["confidence"], 3),
                "reasoning": r["reasoning"],
                "chunks": [{"text": c.text, "score": round(c.score, 4)} for c in r["chunks"]],
            }
            for r in ranked
        ],
        "timing_ms": {**result.get("timing_ms", {}), "total": total_ms},
    }
