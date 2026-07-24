import os
import json
import time
import sqlite3
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from app.config import get_connection
from app.entity_resolver import resolve_form
from app.telemetry import telemetry
from app.cache import query_cache

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3-flash-preview"
TOP_K = 10
SQLITE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_knowledge.db"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            api_key = "dummy_key_for_testing"
        _client = genai.Client(api_key=api_key)
    return _client


def embed_query(text: str):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "dummy_key_for_testing":
        return [0.01] * 768

    try:
        client = get_client()
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding API notice ({e}). Returning fallback vector.")
        return [0.01] * 768


def retrieve_chunks_postgres(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT dc.chunk_text, d.filename, 1 - (dc.embedding <=> %s::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def retrieve_chunks_sqlite(query_embedding, top_k=TOP_K):
    if not SQLITE_DB_PATH.exists():
        return []

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT dc.chunk_text, d.filename, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        results = []
        for text, filename, emb_json in rows:
            emb_vec = np.array(json.loads(emb_json), dtype=np.float32)
            denom = q_norm * np.linalg.norm(emb_vec)
            sim = float(np.dot(q_vec, emb_vec) / denom) if denom > 0 else 0.0
            results.append((text, filename, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    finally:
        cursor.close()
        conn.close()


def retrieve_chunks(query_embedding, top_k=TOP_K):
    try:
        res = retrieve_chunks_postgres(query_embedding, top_k=top_k)
        if res:
            return res
    except Exception:
        pass
    return retrieve_chunks_sqlite(query_embedding, top_k=top_k)


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {filename}]\n{text}" for text, filename, _ in chunks
    )
    return f"""You are an AI assistant providing accurate information about financial policies, loan terms, customer support, and agreements based on internal documentation.
Use ONLY the context below to answer the question. If the answer is not present in the context, clearly state that you do not know or that information is not available in the knowledge base.

RULES:
- Base your answers strictly on the retrieved context below.
- Do not invent rules, fees, or procedural steps not stated in the source documents.
- Always cite the source filename in brackets [Source: filename] next to relevant statements.
- If multiple context chunks provide details, synthesize them into a clear, cohesive answer.

Context:
{context}

Question: {question}

Answer clearly with source citations:"""


def generate_answer(question: str, chunks=None, retries=3, use_cache=True):
    start_time = time.time()

    # Check Cache
    if use_cache:
        cached_res = query_cache.get(question)
        if cached_res:
            latency = (time.time() - start_time) * 1000
            telemetry.record_request(
                question=question,
                latency_ms=latency,
                cached=True,
                input_tokens=cached_res.get("input_tokens", 0),
                output_tokens=cached_res.get("output_tokens", 0),
                retrieved_count=len(cached_res.get("chunks", []))
            )
            cached_res["latency_ms"] = round(latency, 2)
            cached_res["cached"] = True
            return cached_res

    # Retrieve if not cached
    if chunks is None:
        query_embedding = embed_query(question)
        chunks = retrieve_chunks(query_embedding)

    if not chunks:
        latency = (time.time() - start_time) * 1000
        res = {
            "answer": "I could not find relevant information in the knowledge base.",
            "chunks": [],
            "latency_ms": round(latency, 2),
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0
        }
        telemetry.record_request(question, latency, cached=False)
        return res

    prompt = build_prompt(question, chunks)
    answer_text = ""
    input_tokens = len(prompt.split()) * 1.3
    output_tokens = 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "dummy_key_for_testing":
        context_snippets = [f"[{fn}] {text[:150]}" for text, fn, _ in chunks[:3]]
        answer_text = "Context retrieved: " + " | ".join(context_snippets)
    else:
        for attempt in range(retries):
            try:
                client = get_client()
                response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
                answer_text = response.text
                output_tokens = len(answer_text.split()) * 1.3
                break
            except ServerError as e:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    time.sleep(wait)
                    continue
                answer_text = "System is temporarily overloaded. Please try again in a moment."
            except Exception as e:
                print(f"Unexpected error ({e}). Using context summary fallback.")
                context_snippets = [f"[{fn}] {text[:150]}" for text, fn, _ in chunks[:3]]
                answer_text = "Context retrieved: " + " | ".join(context_snippets)

    latency = (time.time() - start_time) * 1000
    telemetry_entry = telemetry.record_request(
        question=question,
        latency_ms=latency,
        cached=False,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        retrieved_count=len(chunks)
    )

    result = {
        "answer": answer_text,
        "chunks": [(text, fn, float(sim)) for text, fn, sim in chunks],
        "latency_ms": round(latency, 2),
        "cached": False,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "estimated_cost_usd": telemetry_entry["estimated_cost_usd"]
    }

    if use_cache and answer_text:
        query_cache.set(question, result)

    return result


def ask(question: str):
    res = generate_answer(question)
    form_match = resolve_form(question)

    print(f"\nQ: {question}")
    if form_match:
        print(f"(Possible related document/form: {form_match['form_name']})")
    print(f"A: {res['answer']}")
    print(f"[Stats: Latency={res['latency_ms']}ms | Cost=${res['estimated_cost_usd']:.6f} | Cached={res['cached']}]\n")
    return res["answer"]


if __name__ == "__main__":
    test_questions = [
        "What is the procedure for early loan repayment?",
        "What is the procedure for early loan repayment?",
        "How are complaints handled according to the policy?",
        "What options are available for financial hardship or job loss?",
    ]
    for q in test_questions:
        ask(q)

    print("\n--- Telemetry Metrics Summary ---")
    print(json.dumps(telemetry.get_metrics(), indent=2))
