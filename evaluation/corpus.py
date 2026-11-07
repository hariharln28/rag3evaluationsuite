from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from .paths import CORPUS_DIR, SUITE_ROOT

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    order: int

    def as_dict(self) -> dict:
        return asdict(self)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc_id: str, title: str, body: str, max_chars: int = 520, overlap: int = 80) -> list[Chunk]:
    paragraphs = _split_paragraphs(body)
    chunks: list[Chunk] = []
    buf = ""
    order = 0

    def flush(piece: str) -> None:
        nonlocal order
        piece = piece.strip()
        if not piece:
            return
        chunks.append(
            Chunk(
                id=f"{doc_id}::{order}",
                doc_id=doc_id,
                title=title,
                text=piece,
                order=order,
            )
        )
        order += 1

    for para in paragraphs:
        if para.startswith("#"):
            para = re.sub(r"^#+\s*", "", para).strip()
            if not buf:
                title = para
            continue
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n\n{para}".strip() if buf else para
            continue
        if buf:
            flush(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
        else:
            for i in range(0, len(para), max_chars - overlap):
                flush(para[i : i + max_chars])
            buf = ""
    if buf:
        flush(buf)
    return chunks


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    docs = []
    if not CORPUS_DIR.exists():
        return docs
    for path in sorted(CORPUS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", raw, re.M)
        docs.append(
            {
                "id": path.stem,
                "title": title_match.group(1).strip() if title_match else path.stem,
                "path": str(path),
                "text": raw,
            }
        )
    return docs


@lru_cache(maxsize=1)
def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents():
        chunks.extend(chunk_document(doc["id"], doc["title"], doc["text"]))
    return chunks
