import sys
import json
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main import generate_answer, embed_query, retrieve_chunks

EVAL_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"

# Ground Truth Evaluation Dataset
EVAL_DATASET = [
    {
        "id": 1,
        "query": "What is the procedure for early loan repayment?",
        "target_doc": "early_repayment.txt",
        "expected_keywords": ["bondora.fi", "repay early", "PIN", "repaid in full"]
    },
    {
        "id": 2,
        "query": "How are customer complaints handled according to the policy?",
        "target_doc": "Complaints-procedure.pdf",
        "expected_keywords": ["Complaints", "5 business days", "15 days", "Consumer Disputes Board"]
    },
    {
        "id": 3,
        "query": "What support is available during financial hardship or job loss?",
        "target_doc": "financial_hardship.txt",
        "expected_keywords": ["financial hardship", "payment holiday", "payslips", "reschedule"]
    },
    {
        "id": 4,
        "query": "What is B-Secure and what benefits does it offer?",
        "target_doc": "what_is_b_secure.txt",
        "expected_keywords": ["B-Secure", "10", "restructure", "principal payment holiday"]
    },
    {
        "id": 5,
        "query": "What are the rules for debt collection?",
        "target_doc": "debt_collection_process.txt",
        "expected_keywords": ["debt", "collection"]
    },
    {
        "id": 6,
        "query": "What are the total loan amounts and average interest rates by country in the Bondora dataset?",
        "target_doc": "bondora_loan_dataset_summary.txt",
        "expected_keywords": ["Estonia", "Finland", "Spain", "25.32%"]
    },
    {
        "id": 7,
        "query": "How do I change my monthly payment date?",
        "target_doc": "change_payment_date.txt",
        "expected_keywords": ["payment date", "1st and 27th"]
    },
    {
        "id": 8,
        "query": "What is the process for closing an account?",
        "target_doc": "closing_account.txt",
        "expected_keywords": ["closing", "account"]
    },
    {
        "id": 9,
        "query": "What terms apply to automated payments?",
        "target_doc": "Terms-and-conditions-for-automated-payments.pdf",
        "expected_keywords": ["automated", "payments"]
    },
    {
        "id": 10,
        "query": "Where can I find the conflict policy information?",
        "target_doc": "Conflict-policy.pdf",
        "expected_keywords": ["Conflict", "policy"]
    }
]


def run_evaluation():
    print("Starting automated RAG evaluation benchmark run...")
    hits = 0
    faithfulness_scores = []
    latencies = []
    costs = []

    results = []

    for item in EVAL_DATASET:
        q = item["query"]
        target = item["target_doc"]
        keywords = item["expected_keywords"]

        print(f"Evaluating Q{item['id']}: '{q[:40]}...'")

        # Run query (bypassing cache for benchmark accuracy)
        res = generate_answer(q, use_cache=False)
        latencies.append(res["latency_ms"])
        costs.append(res["estimated_cost_usd"])

        # Check Hit@10
        retrieved_docs = [fn for _, fn, _ in res["chunks"]]
        hit = any(target.lower() in fn.lower() for fn in retrieved_docs)
        if hit:
            hits += 1

        # Check Faithfulness / Keyword Coverage
        ans = res["answer"]
        matched_kw = [kw for kw in keywords if kw.lower() in ans.lower()]
        faithfulness = len(matched_kw) / len(keywords) if keywords else 1.0
        faithfulness_scores.append(faithfulness)

        results.append({
            "id": item["id"],
            "query": q,
            "target_doc": target,
            "hit_at_10": hit,
            "faithfulness_score": round(faithfulness, 2),
            "matched_keywords": matched_kw,
            "latency_ms": res["latency_ms"],
            "cost_usd": res["estimated_cost_usd"],
            "retrieved_count": len(retrieved_docs)
        })
        time.sleep(0.5)

    hit_at_10 = round(hits / len(EVAL_DATASET), 2)
    avg_faithfulness = round(float(np.mean(faithfulness_scores)), 2)
    p50_latency = round(float(np.percentile(latencies, 50)), 2)
    p95_latency = round(float(np.percentile(latencies, 95)), 2)
    avg_cost = round(float(np.mean(costs)), 6)
    total_cost_1k = round(avg_cost * 1000, 4)

    summary = {
        "timestamp": time.time(),
        "total_test_cases": len(EVAL_DATASET),
        "hit_at_10_recall": hit_at_10,
        "avg_faithfulness_score": avg_faithfulness,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "avg_cost_per_query_usd": avg_cost,
        "estimated_cost_per_1k_queries_usd": total_cost_1k,
        "test_results": results
    }

    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION BENCHMARK RESULTS ================")
    print(f" Hit@10 Recall Rate:       {hit_at_10 * 100:.1f}%")
    print(f" Avg Faithfulness Score:  {avg_faithfulness * 100:.1f}%")
    print(f" p50 Latency:             {p50_latency} ms")
    print(f" p95 Latency:             {p95_latency} ms")
    print(f" Avg Cost / Request:      ${avg_cost:.6f}")
    print(f" Cost / 1,000 Queries:    ${total_cost_1k:.4f}")
    print("==============================================================")
    print(f"Saved evaluation results to {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
