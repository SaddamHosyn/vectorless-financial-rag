import time
import hashlib
from typing import Optional, Dict, Any

class QueryResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _hash_key(self, text: str) -> str:
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(question)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, question: str, response_data: Dict[str, Any]) -> None:
        key = self._hash_key(question)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": response_data
        }

    def clear(self) -> None:
        self.cache.clear()

# Global singleton cache instance
query_cache = QueryResponseCache()
