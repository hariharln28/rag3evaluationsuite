from __future__ import annotations

import io
import csv
import pathlib
import sys
from typing import Any, Optional

# Ensure project root is in sys.path for serverless execution
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import APIRouter, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evaluation.corpus import load_chunks, load_documents
from evaluation.evaluate import evaluate_item, load_gold, run_suite, summarize
from evaluation.generation import generate, is_abstain
from evaluation.retrieval import get_retriever

app = FastAPI(
    title="RAG3 Evaluation Suite API",
    description="Benchmarking and evaluation for RAG retrieval quality vs generation quality",
    version="1.0.0",
)

# Enable CORS for local development and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter()


class EvaluateParams(BaseModel):
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid")
    abstain_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=15)
    mode: str = Field(default="hybrid")
    abstain_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


@api.get("/health")
def health():
    return {"status": "ok", "service": "rag3evaluationsuite", "version": "1.0.0"}


@api.get("/summary")
def get_summary(
    k: int = Query(5, ge=1, le=20),
    mode: str = Query("hybrid"),
    abstain_threshold: float = Query(0.35, ge=0.0, le=1.0),
):
    try:
        suite = run_suite(k=k, mode=mode, abstain_threshold=abstain_threshold)
        return suite
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/evaluate")
def post_evaluate(params: EvaluateParams):
    try:
        suite = run_suite(k=params.k, mode=params.mode, abstain_threshold=params.abstain_threshold)
        return suite
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retriever = get_retriever()
    retrieved = retriever.search(q, k=req.k, mode=req.mode)
    gen = generate(q, retrieved, abstain_threshold=req.abstain_threshold)
    abstained = gen.abstained or is_abstain(gen.answer)

    chunks_data = [
        {
            "rank": r.rank,
            "chunk_id": r.chunk.id,
            "doc_id": r.chunk.doc_id,
            "title": r.chunk.title,
            "text": r.chunk.text,
            "score": round(r.score, 4),
            "bm25": round(r.bm25, 4),
            "tfidf": round(r.tfidf, 4),
        }
        for r in retrieved
    ]

    return {
        "question": q,
        "mode": req.mode,
        "k": req.k,
        "answer": gen.answer,
        "abstained": abstained,
        "evidence_chunk_id": gen.evidence_chunk_id,
        "evidence_sentence": gen.evidence_sentence,
        "confidence": round(gen.confidence, 4),
        "retrieved_count": len(chunks_data),
        "chunks": chunks_data,
    }


@api.get("/gold")
def get_gold_questions():
    gold = load_gold()
    return {"count": len(gold), "questions": gold}


@api.get("/corpus")
def get_corpus():
    docs = load_documents()
    summary = []
    for d in docs:
        summary.append(
            {
                "id": d["id"],
                "title": d["title"],
                "char_length": len(d["text"]),
                "snippet": d["text"][:240].replace("\n", " ").strip() + "...",
            }
        )
    return {"count": len(summary), "documents": summary}


@api.get("/corpus/{doc_id}")
def get_corpus_doc(doc_id: str):
    docs = load_documents()
    for d in docs:
        if d["id"] == doc_id:
            return d
    raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


@api.get("/export.csv")
def export_csv(
    k: int = Query(5, ge=1, le=20),
    mode: str = Query("hybrid"),
    abstain_threshold: float = Query(0.35, ge=0.0, le=1.0),
):
    suite = run_suite(k=k, mode=mode, abstain_threshold=abstain_threshold)
    results = suite["results"]

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "id", "question", "category", "difficulty", "answerable",
        "retrieval_hit", "retrieval_rank", "oracle_correct", "oracle_f1",
        "rag_correct", "rag_f1", "failure_class", "gold_answer", "model_answer"
    ])
    for r in results:
        writer.writerow([
            r["id"],
            r["question"],
            r["category"],
            r["difficulty"],
            r["answerable"],
            r["retrieval"]["hit"],
            r["retrieval"]["rank"] or "",
            r["oracle_generation"]["correct"],
            r["oracle_generation"]["token_f1"],
            r["rag_generation"]["correct"],
            r["rag_generation"]["token_f1"],
            r["failure_class"],
            r["gold_answer"] or "<unanswerable>",
            r["rag_generation"]["answer"],
        ])

    return Response(
        content=out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rag3_evaluation_results.csv"},
    )


# Mount router under /api AND at root for Vercel prefix flexibility
app.include_router(api, prefix="/api")
app.include_router(api)

# Static file serving for local execution
PUBLIC_DIR = ROOT_DIR / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

    @app.get("/")
    def read_root():
        index_file = PUBLIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "ok", "message": "RAG3 Evaluation Suite"}
