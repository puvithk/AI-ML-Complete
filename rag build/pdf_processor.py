"""
PDF -> Sections -> Chunks

Detects section boundaries in a PDF using font-size / boldness heuristics
(falls back to numbered-heading regex), assigns each section a priority,
then splits each section into overlapping sub-chunks for embedding.
"""
import re
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List


HEADING_KEYWORDS_HIGH = [
    "abstract", "conclusion", "summary", "results", "executive summary",
    "key findings", "recommendation", "overview",
]
HEADING_KEYWORDS_LOW = [
    "appendix", "references", "bibliography", "acknowledg", "disclaimer",
]

NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*)[\.\)]?\s+[A-Z][^\n]{2,80}$")


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    text: str
    order: int


@dataclass
class Section:
    section_id: str
    title: str
    text: str
    priority: float  # 0.0 - 1.0, higher = more important
    order: int
    chunks: List[Chunk] = field(default_factory=list)


def _extract_blocks_with_style(pdf_bytes: bytes):
    """Return list of (text, font_size, is_bold, page_no) per line/span."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks = []
    for page_no, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text:
                    continue
                sizes = [s["size"] for s in line["spans"]]
                bold = any((s["flags"] & 2**4) for s in line["spans"])
                blocks.append((line_text, max(sizes), bold, page_no))
    doc.close()
    return blocks


def _looks_like_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    if len(text) > 90:
        return False
    if NUMBERED_HEADING_RE.match(text):
        return True
    if size >= body_size + 1.5 and len(text.split()) <= 12:
        return True
    if bold and size >= body_size and len(text.split()) <= 10 and text[0:1].isupper():
        return True
    return False


def _score_priority(title: str) -> float:
    t = title.lower()
    if any(k in t for k in HEADING_KEYWORDS_HIGH):
        return 0.9
    if any(k in t for k in HEADING_KEYWORDS_LOW):
        return 0.25
    return 0.6  # default/neutral priority


def _split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Sentence-aware sliding window chunking."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 20]


def process_pdf(pdf_bytes: bytes, doc_id: str) -> List[Section]:
    blocks = _extract_blocks_with_style(pdf_bytes)
    if not blocks:
        return []

    sizes = sorted(b[1] for b in blocks)
    body_size = sizes[len(sizes) // 2] if sizes else 10.0

    raw_sections = []
    current_title, current_lines = "Introduction", []
    for text, size, bold, page_no in blocks:
        if _looks_like_heading(text, size, bold, body_size):
            if current_lines:
                raw_sections.append((current_title, " ".join(current_lines)))
            current_title, current_lines = text, []
        else:
            current_lines.append(text)
    if current_lines:
        raw_sections.append((current_title, " ".join(current_lines)))

    merged = []
    for title, body in raw_sections:
        if len(body) < 60 and merged:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body} {title}. {body}")
        else:
            merged.append((title, body))

    sections: List[Section] = []
    for i, (title, body) in enumerate(merged):
        if not body.strip():
            continue
        section_id = f"{doc_id}::sec{i}"
        section = Section(
            section_id=section_id,
            title=title.strip()[:120],
            text=body.strip(),
            priority=_score_priority(title),
            order=i,
        )
        for j, chunk_text in enumerate(_split_into_chunks(body)):
            section.chunks.append(
                Chunk(chunk_id=f"{section_id}::c{j}", section_id=section_id, text=chunk_text, order=j)
            )
        if section.chunks:
            sections.append(section)

    return sections
