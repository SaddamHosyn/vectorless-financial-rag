import time
import numpy as np
from typing import Dict, Any, List

class TelemetryTracker:
    def __init__(self):
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.latencies: List[float] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.history: List[Dict[str, Any]] = []

    def record_request(
        self,
        question: str,
        latency_ms: float,
        cached: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retrieved_count: int = 0
    ) -> Dict[str, Any]:
        self.total_requests += 1
        if cached:
            self.cache_hits += 1
        self.latencies.append(latency_ms)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Gemini pricing estimate: $0.075/1M input, $0.30/1M output, $0.02/1M embedding
        cost = (input_tokens * 0.000000075) + (output_tokens * 0.00000030)
        if not cached:
            cost += (len(question.split()) * 1.3 * 0.00000002)

        entry = {
            "timestamp": time.time(),
            "question": question,
            "latency_ms": round(latency_ms, 2),
            "cached": cached,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "retrieved_count": retrieved_count
        }
        self.history.append(entry)
        return entry

    def get_metrics(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "total_requests": 0,
                "cache_hit_rate_pct": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "total_estimated_cost_usd": 0.0,
                "avg_cost_per_query_usd": 0.0
            }

        lats = np.array(self.latencies)
        cache_rate = (self.cache_hits / self.total_requests) * 100 if self.total_requests > 0 else 0.0
        
        # Calculate total cost across history
        total_cost = sum(h["estimated_cost_usd"] for h in self.history)
        avg_cost = total_cost / self.total_requests if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_pct": round(cache_rate, 2),
            "p50_latency_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost_usd": round(total_cost, 6),
            "avg_cost_per_query_usd": round(avg_cost, 6)
        }

# Global singleton telemetry instance
telemetry = TelemetryTracker()
