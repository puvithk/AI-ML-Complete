"""
FastAPI endpoint for the database agent.

Run from inside the `database_agent/` directory (so the `agent.*` / `utils.*`
imports resolve):

    uvicorn api:app --port 8000

Then call it:

    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "Find the top 5 record based on the bill amount",
              "session_id": "demo"}'

This file also wires a small *relevant* long-term memory onto the agent. The
graph keeps short-term (thread) state via its checkpointer; what was missing
(see README) was carrying forward only the useful facts across turns. We
deliberately do NOT store every message - greetings, failures and empty
answers are dropped, and on each turn only the memories that actually overlap
the new question are injected. No nonsense memory.
"""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.main_agent.graph import agent
from utils.config import get_settings
from utils.database import check_health, close_pool
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)
settings = get_settings()


# --------------------------------------------------------------------------- #
# Relevant memory                                                             #
# --------------------------------------------------------------------------- #
# One store shared across requests, keyed by session_id. Only meaningful
# question -> answer pairs are kept, and recall returns only the entries that
# are actually relevant to the incoming question.

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "with", "by", "at", "as", "it", "this", "that", "from",
    "what", "which", "who", "how", "many", "much", "give", "me", "show", "find",
    "list", "get", "all", "based", "top", "record", "records", "please", "can",
    "you", "do", "does", "i", "we", "our", "my",
}

# Answers that carry no reusable information - do not remember these.
_NON_ANSWER_RE = re.compile(
    r"\b(no data|not available|unavailable|no result|couldn't|could not|"
    r"unable|i don't|i do not|error|failed|no information|cannot|no matching)\b",
    re.IGNORECASE,
)

# Pure chit-chat that should never enter memory.
_GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank you|bye|goodbye|ok|okay)\b[\s!.?]*$",
    re.IGNORECASE,
)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


class RelevantMemory:
    """Session-scoped store that keeps only relevant question/answer facts.

    Bounded on two axes so it can't grow without limit in a long-running
    process: `max_per_session` entries per session, and `max_sessions`
    sessions total (least-recently-used session evicted).
    """

    def __init__(self, max_per_session: int, recall_k: int, max_sessions: int) -> None:
        self._store: "OrderedDict[str, list[dict]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_per_session = max_per_session
        self._recall_k = recall_k
        self._max_sessions = max_sessions

    def is_worth_remembering(self, question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return False
        if _GREETING_RE.match(question or ""):
            return False
        if _NON_ANSWER_RE.search(answer):
            return False
        # Needs at least one content keyword to be recallable later.
        return bool(_keywords(question))

    def remember(self, session_id: str, question: str, answer: str) -> bool:
        if not self.is_worth_remembering(question, answer):
            return False
        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "keywords": _keywords(question) | _keywords(answer),
        }
        with self._lock:
            bucket = self._store.get(session_id)
            if bucket is None:
                bucket = []
                self._store[session_id] = bucket
            self._store.move_to_end(session_id)
            bucket.append(entry)
            if len(bucket) > self._max_per_session:
                del bucket[: len(bucket) - self._max_per_session]
            # Evict least-recently-used sessions.
            while len(self._store) > self._max_sessions:
                self._store.popitem(last=False)
        return True

    def recall(self, session_id: str, question: str) -> list[dict]:
        """Return only the stored entries relevant to `question`."""
        q_keywords = _keywords(question)
        if not q_keywords:
            return []
        with self._lock:
            bucket = list(self._store.get(session_id, []))

        scored = []
        for entry in bucket:
            overlap = len(q_keywords & entry["keywords"])
            if overlap > 0:
                scored.append((overlap, entry))
        # Most relevant first; ties keep insertion (recency-ish) order.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[: self._recall_k]]

    def dump(self, session_id: str) -> list[dict]:
        with self._lock:
            return [
                {"question": e["question"], "answer": e["answer"]}
                for e in self._store.get(session_id, [])
            ]

    def clear(self, session_id: str) -> int:
        with self._lock:
            removed = len(self._store.get(session_id, []))
            self._store.pop(session_id, None)
        return removed


memory = RelevantMemory(
    max_per_session=settings.memory_max_per_session,
    recall_k=settings.memory_recall_k,
    max_sessions=settings.memory_max_sessions,
)


def _build_question_with_memory(session_id: str, question: str) -> str:
    """Prepend only the relevant recalled facts to the user's question."""
    relevant = memory.recall(session_id, question)
    if not relevant:
        return question
    lines = [f"- Q: {e['question']}\n  A: {e['answer']}" for e in relevant]
    context = "\n".join(lines)
    return (
        "Relevant facts from earlier in this conversation "
        "(use only if helpful):\n"
        f"{context}\n\n"
        f"Current question: {question}"
    )


def _extract_answer(result: dict) -> str:
    """Pull the user-facing answer out of the graph's final state."""
    summarized = (result or {}).get("summarized_output")
    if isinstance(summarized, str) and summarized.strip():
        return summarized.strip()

    final = (result or {}).get("final_answer")
    if isinstance(final, list):
        parts = [str(p).strip() for p in final if str(p).strip()]
        return "\n".join(parts)
    if isinstance(final, str):
        return final.strip()
    return ""


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session_id format.")


# --------------------------------------------------------------------------- #
# FastAPI app                                                                 #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Database Agent API starting up.")
    yield
    logger.info("Database Agent API shutting down; closing DB pool.")
    close_pool()


app = FastAPI(title="Database Agent API", version="1.0.0", lifespan=lifespan)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Natural-language question")
    session_id: str = Field("default", max_length=128,
                            description="Conversation/thread id")


class AskResponse(BaseModel):
    session_id: str
    answer: str
    decomposed_questions: list[str] = []
    used_memory: list[dict] = []
    remembered: bool


@app.get("/health")
def health() -> dict:
    db_ok = check_health()
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    _validate_session_id(req.session_id)

    used_memory = memory.recall(req.session_id, req.question)
    question_for_agent = _build_question_with_memory(req.session_id, req.question)

    # thread_id gives the graph its checkpoint continuity for this session.
    # The SQL subgraph borrows its own pooled DB connection per branch.
    config = {"configurable": {"thread_id": req.session_id}}
    try:
        result = agent.invoke({"user_question": question_for_agent}, config=config)
    except Exception:
        # Log the full error internally; return a sanitized, correlatable
        # message to the caller (never leak internals/DB details).
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Agent invocation failed [error_id=%s]", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"The request could not be completed. Reference: {error_id}",
        )

    answer = _extract_answer(result)
    remembered = memory.remember(req.session_id, req.question, answer)

    return AskResponse(
        session_id=req.session_id,
        answer=answer or "No answer could be produced for this question.",
        decomposed_questions=(result or {}).get("decomposed_question", []) or [],
        used_memory=[
            {"question": e["question"], "answer": e["answer"]} for e in used_memory
        ],
        remembered=remembered,
    )


@app.get("/memory/{session_id}")
def get_memory(session_id: str) -> dict:
    _validate_session_id(session_id)
    return {"session_id": session_id, "memory": memory.dump(session_id)}


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str) -> dict:
    _validate_session_id(session_id)
    removed = memory.clear(session_id)
    return {"session_id": session_id, "cleared": removed}
