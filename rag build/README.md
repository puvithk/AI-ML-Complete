# Section-Parallel RAG Agent

A RAG pipeline that:
1. Splits a large PDF into **sections** (using font-size/bold heading detection + numbered-heading regex).
2. Assigns each section a **priority** (e.g. "Results"/"Conclusion" > generic body > "Appendix"/"References").
3. Splits each section into overlapping **sub-chunks**, indexed **separately per section** (no cross-section vocab bleed).
4. On query, runs retrieval across **all sections concurrently** via a LangGraph `Send` fan-out (map step), then a fan-in synthesis step that weights each section's result by `relevance × priority`.
5. Optionally calls Claude to write a final cited answer from the top-ranked sections; without an API key it falls back to the best extractive match (so the retrieval path always works standalone).

## Why it's low-latency
- Retrieval uses per-section **TF-IDF + cosine similarity** (scikit-learn) instead of a neural embedder — no model download/load, millisecond-scale search, small vocab per section.
- All sections are searched **in parallel** (LangGraph `Send`), not sequentially.
- Only one LLM call happens per query (final synthesis), not one per section — keeps network latency bounded regardless of document size.
- The final synthesis prompt is capped to the top `MAX_SECTIONS_TO_LLM` (default 6) ranked sections, so prompt size (and therefore latency/cost) doesn't grow with document size.

## Run it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then just open `frontend/index.html` in a browser (double-click it, no build step). Point the "Backend URL" field at `http://localhost:8000` (default), upload a PDF, and ask a question. Paste an Anthropic API key in the frontend if you want synthesized (rather than extractive) answers — it's only sent to your own backend, never stored.

## Structure
```
backend/
  pdf_processor.py   -> PDF -> sections -> chunks
  vector_store.py     -> per-section TF-IDF index
  agent.py             -> LangGraph parallel fan-out + priority-weighted synthesis
  main.py               -> FastAPI /upload and /query endpoints
frontend/
  index.html             -> single-file UI (upload, live section priorities, parallel per-section results, final answer)
```

## Notes / next steps if you want to extend it
- Swap TF-IDF for a real embedder (e.g. a local `sentence-transformers` model or an API embedding call) if you need semantic (not lexical) matching — the `SectionIndex` class is the only place that would need to change.
- `DocumentStore` is in-memory/per-process; swap in Redis or a vector DB (per-section namespace/collection) for multi-instance deployments.
- Section detection is heuristic (font size + bold + numbered headings). For very irregularly formatted PDFs you could add an LLM-based heading classifier as a fallback.
