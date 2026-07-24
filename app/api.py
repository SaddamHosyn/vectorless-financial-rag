import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.main import generate_answer
from app.telemetry import telemetry

app = FastAPI(
    title="Financial Policy RAG Engine",
    description="Production-grade Retrieval-Augmented Generation API with pgvector storage, sub-10ms response caching, and MLOps observability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"


class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the procedure for early loan repayment?")
    top_k: Optional[int] = Field(default=10, example=10)
    use_cache: Optional[bool] = Field(default=True, example=True)


class ChunkCitation(BaseModel):
    filename: str
    similarity: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[ChunkCitation]
    latency_ms: float
    cached: bool
    estimated_cost_usd: float


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "financial-rag-api",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG Inference"])
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    res = generate_answer(req.question, use_cache=req.use_cache)
    
    citations = [
        ChunkCitation(
            filename=fn,
            similarity=round(sim, 4),
            snippet=text[:250] + ("..." if len(text) > 250 else "")
        )
        for text, fn, sim in res.get("chunks", [])
    ]

    return QueryResponse(
        question=req.question,
        answer=res["answer"],
        citations=citations,
        latency_ms=res["latency_ms"],
        cached=res["cached"],
        estimated_cost_usd=res["estimated_cost_usd"]
    )


@app.get("/metrics", tags=["Observability"])
def get_telemetry_metrics():
    return telemetry.get_metrics()


@app.get("/eval", tags=["MLOps Evaluation"])
def get_evaluation_results():
    if not EVAL_FILE_PATH.exists():
        return {
            "status": "pending",
            "message": "Evaluation suite has not been executed yet. Run python scripts/evaluate_rag.py"
        }
    try:
        with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {e}")
