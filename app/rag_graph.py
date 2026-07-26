"""
app/rag_graph.py

LangGraph Stateful RAG Workflow Graph
======================================
Nodes:
  1. cache_check_node      → Check in-memory TTL cache first
  2. embed_retrieve_node   → Embed query + 2-Stage HF Cross-Encoder retrieval
  3. grade_context_node    → Evaluate retrieved chunk quality (relevance gate)
  4. generate_node         → Gemini LLM answer generation with source citations
  5. telemetry_node        → Record latency, tokens, cost to MLOps tracker

Routing Edges:
  cache_check → CACHE_HIT  → telemetry → END
  cache_check → CACHE_MISS → embed_retrieve
  grade_context → GOOD     → generate
  grade_context → LOW      → generate  (with low_recall flag for fallback context)
  generate     → telemetry → END
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHARED STATE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class RAGState(TypedDict, total=False):
    # Input
    question: str
    use_cache: bool
    top_k: int

    # Intermediate
    query_embedding: List[float]
    chunks: List[Tuple[str, str, float]]   # (text, filename, score)
    context_quality: str                   # "good" | "low"
    low_recall: bool
    prompt: str

    # Output
    answer: str
    cached: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float

    # Internal timing
    _start_time: float


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cache_check_node(state: RAGState) -> RAGState:
    """Node 1: Check in-memory TTL cache before any model calls."""
    from app.cache import query_cache

    state["_start_time"] = time.time()
    question = state["question"]
    use_cache = state.get("use_cache", True)

    if use_cache:
        cached = query_cache.get(question)
        if cached:
            state["answer"] = cached.get("answer", "")
            state["chunks"] = cached.get("chunks", [])
            state["input_tokens"] = cached.get("input_tokens", 0)
            state["output_tokens"] = cached.get("output_tokens", 0)
            state["estimated_cost_usd"] = cached.get("estimated_cost_usd", 0.0)
            state["cached"] = True
            return state

    state["cached"] = False
    return state


def embed_retrieve_node(state: RAGState) -> RAGState:
    """Node 2: Embed query + 2-Stage Cross-Encoder retrieval."""
    from app.main import embed_query, retrieve_chunks

    question = state["question"]
    top_k = state.get("top_k", 10)

    embedding = embed_query(question)
    chunks = retrieve_chunks(embedding, top_k=top_k, query=question)

    state["query_embedding"] = list(embedding)
    state["chunks"] = chunks
    return state


def grade_context_node(state: RAGState) -> RAGState:
    """
    Node 3: Context Quality Gate.
    Grades the top retrieved chunk's relevance score.
    Score > 0.30  → 'good'  (proceed to full LLM generation)
    Score <= 0.30 → 'low'   (proceed with low_recall flag so prompt warns LLM)
    """
    chunks = state.get("chunks", [])

    if not chunks:
        state["context_quality"] = "low"
        state["low_recall"] = True
        return state

    # Check top Cross-Encoder / cosine score
    top_score = chunks[0][2] if chunks else 0.0

    if top_score > 0.30:
        state["context_quality"] = "good"
        state["low_recall"] = False
    else:
        state["context_quality"] = "low"
        state["low_recall"] = True

    return state


def generate_node(state: RAGState) -> RAGState:
    """Node 4: Build prompt and generate answer with Gemini LLM."""
    import os
    from google import genai
    from google.genai.errors import ServerError
    from app.main import build_prompt, get_client, GEN_MODEL

    question = state["question"]
    chunks = state.get("chunks", [])
    low_recall = state.get("low_recall", False)

    if not chunks:
        state["answer"] = "I could not find relevant information in the knowledge base."
        state["input_tokens"] = 0
        state["output_tokens"] = 0
        state["estimated_cost_usd"] = 0.0
        return state

    prompt = build_prompt(question, chunks)
    if low_recall:
        prompt = (
            "[Note: Retrieved context has low confidence. "
            "Answer only from the context below; admit if information is insufficient.]\n\n"
            + prompt
        )

    answer_text = ""
    input_tokens = int(len(prompt.split()) * 1.3)
    output_tokens = 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "dummy_key_for_testing":
        snippets = [f"[{fn}] {text[:150]}" for text, fn, _ in chunks[:3]]
        answer_text = "Context retrieved: " + " | ".join(snippets)
    else:
        for attempt in range(3):
            try:
                client = get_client()
                response = client.models.generate_content(
                    model=GEN_MODEL, contents=prompt
                )
                answer_text = response.text
                output_tokens = int(len(answer_text.split()) * 1.3)
                break
            except ServerError:
                if attempt < 2:
                    import time as _t
                    _t.sleep(5 * (attempt + 1))
                    continue
                answer_text = "System is temporarily overloaded. Please try again."
            except Exception as exc:
                snippets = [f"[{fn}] {text[:150]}" for text, fn, _ in chunks[:3]]
                answer_text = "Context retrieved: " + " | ".join(snippets)
                break

    state["answer"] = answer_text
    state["input_tokens"] = input_tokens
    state["output_tokens"] = output_tokens
    return state


def telemetry_node(state: RAGState) -> RAGState:
    """Node 5: Record request to MLOps telemetry tracker and write to cache."""
    from app.telemetry import telemetry
    from app.cache import query_cache

    start_time = state.get("_start_time", time.time())
    latency_ms = (time.time() - start_time) * 1000
    question = state["question"]
    cached = state.get("cached", False)
    chunks = state.get("chunks", [])
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)

    entry = telemetry.record_request(
        question=question,
        latency_ms=latency_ms,
        cached=cached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retrieved_count=len(chunks),
    )

    state["latency_ms"] = round(latency_ms, 2)
    state["estimated_cost_usd"] = entry["estimated_cost_usd"]

    # Write to cache for future hits
    use_cache = state.get("use_cache", True)
    answer = state.get("answer", "")
    if use_cache and answer and not cached:
        query_cache.set(question, {
            "answer": answer,
            "chunks": chunks,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": entry["estimated_cost_usd"],
        })

    return state


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONDITIONAL ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_cache(state: RAGState) -> str:
    """After cache_check: if cached, skip straight to telemetry; else retrieve."""
    return "telemetry" if state.get("cached") else "embed_retrieve"


def route_after_grading(state: RAGState) -> str:
    """After grade_context: always generate regardless of quality (flag is set)."""
    return "generate"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("cache_check",   cache_check_node)
    graph.add_node("embed_retrieve", embed_retrieve_node)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("generate",      generate_node)
    graph.add_node("telemetry",     telemetry_node)

    # Entry point
    graph.set_entry_point("cache_check")

    # Conditional routing after cache check
    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {"telemetry": "telemetry", "embed_retrieve": "embed_retrieve"},
    )

    # Linear edges for retrieval path
    graph.add_edge("embed_retrieve", "grade_context")

    # Conditional routing after grading (always generates, but state carries quality flag)
    graph.add_conditional_edges(
        "grade_context",
        route_after_grading,
        {"generate": "generate"},
    )

    # Generate → Telemetry → END
    graph.add_edge("generate", "telemetry")
    graph.add_edge("telemetry", END)

    return graph.compile()


# Singleton compiled graph (lazy-initialized on first request)
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_graph(question: str, use_cache: bool = True, top_k: int = 10) -> Dict[str, Any]:
    """
    Execute the full LangGraph RAG workflow and return a result dict
    with the same schema as the previous generate_answer() function.
    """
    graph = get_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "use_cache": use_cache,
        "top_k": top_k,
    }

    final_state = graph.invoke(initial_state)

    chunks = final_state.get("chunks", [])
    return {
        "answer": final_state.get("answer", ""),
        "chunks": [(text, fn, float(sim)) for text, fn, sim in chunks],
        "latency_ms": final_state.get("latency_ms", 0.0),
        "cached": final_state.get("cached", False),
        "input_tokens": final_state.get("input_tokens", 0),
        "output_tokens": final_state.get("output_tokens", 0),
        "estimated_cost_usd": final_state.get("estimated_cost_usd", 0.0),
    }
