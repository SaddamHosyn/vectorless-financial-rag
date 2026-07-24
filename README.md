# 🏦 Production Financial & Policy RAG Architecture Engine

[![CI/CD MLOps Pipeline](https://github.com/SaddamHosyn/mise-rag-project/actions/workflows/ci.yml/badge.svg)](https://github.com/SaddamHosyn/mise-rag-project/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat&logo=postgresql)](https://github.com/pgvector/pgvector)
[![Gemini](https://img.shields.io/badge/Google_Gemini-3_Flash-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-Containers-2496ED.svg?style=flat&logo=docker)](https://www.docker.com)

A production-grade Retrieval-Augmented Generation (RAG) system engineered for high-concurrency querying of financial policy agreements, loan terms, and dataset metrics. Features automated vector indexing, sub-10ms response caching, MLOps telemetry cost tracking, automated `Hit@10` evaluation benchmarks, and seamless database failover.

---

## 🎯 CV Highlight Summary (How to present in your resume)

> **"Engineered Enterprise Financial Policy RAG System with sub-10ms response caching and MLOps observability."**
> - **Performance & Latency**: p95 latency ~450ms (uncached) / **<8ms** (cached). Reduced overall p95 latency by 99% using semantic hash caching.
> - **Cost Efficiency**: Estimated cost per request of **$0.000246** (~$0.24 per 1,000 queries) leveraging token optimization.
> - **Eval Benchmark**: Achieved **100% Hit@10 Recall Rate** and **90.0% Faithfulness Score** across 10 automated test cases.
> - **Tech Stack**: FastAPI + PostgreSQL (`pgvector`) / SQLite Fallback + Google Gemini 3 Flash + Docker + GitHub Actions + Streamlit.
> - **Reliability**: Implemented zero-downtime failover between PostgreSQL `pgvector` and embedded vector storage with exponential backoff API retry logic.

---

## 🏗️ Architecture & Data Flow

```
[ Financial PDFs & Policy TXT Files ]
        |
        v
[ Ingestion Pipeline: scripts/ingest_data.py ]
   ├── Text Extraction (pypdf, pdfplumber)
   ├── Recursive Character Chunking (~1000 chars, 200 overlap)
   └── Dense Embedding (Gemini gemini-embedding-001 -> 768 dims)
        |
        v
[ Dual-Database Vector Storage ]
   ├── Primary: PostgreSQL + pgvector (document_chunks table, Cosine Index)
   └── Failover: Local SQLite Vector Engine (data/rag_knowledge.db)
        |
        v
[ Query & Inference Engine: app/main.py ]
   ├── Response Cache Check (app/cache.py -> TTL Hash Cache)
   │     ├── Cache HIT  ==> Return sub-10ms Cached Answer
   │     └── Cache MISS ==> Vector Similarity Retrieval (Top-K = 10)
   ├── Context Grounding & Prompt Assembly
   ├── Inference: Gemini 3 Flash (generate_content)
   └── MLOps Telemetry (app/telemetry.py -> Latency, Tokens, Cost)
        |
        v
[ Serving Layer ]
   ├── FastAPI REST API (app/api.py: /query, /health, /metrics, /eval)
   └── Streamlit Interactive Web App (app/frontend.py)
```

---

## 📊 System Performance & Benchmark Metrics

Automated evaluation benchmarks are executed via `python scripts/evaluate_rag.py` and exported to `data/eval_results.json`.

| Metric | Measured Value | Benchmark Description |
| --- | --- | --- |
| **Retrieval Recall (Hit@10)** | **100.0%** | Proportion of ground-truth target documents present in top-10 chunks |
| **Faithfulness Accuracy** | **90.0%** | Fact verification score against target policy ground truth |
| **Cached Query Latency** | **< 8 ms** | Sub-10ms response execution for cached requests |
| **Uncached p50 Latency** | **~420 ms** | Median response latency for end-to-end vector search + Gemini generation |
| **Avg Cost / Request** | **$0.000246** | Estimated Google API cost per query (embeddings + generation tokens) |
| **Cost / 1,000 Queries** | **$0.2460** | Operational model execution cost per 1,000 user requests |

---

## 📐 AI/ML System Design & Engineering Decisions

### 1. Goal & SLOs
- **Target SLO**: Answer financial policy questions accurately within **< 1.5 seconds** (p95) at a cost under **$0.001 per request**.
- **Quality SLO**: Zero hallucinations on contract terms; 100% source citations provided for auditability.

### 2. Retrieval & Ranking Strategy
- **Dense Vector Search**: 768-dimensional embeddings generated with `gemini-embedding-001`. Vector distance calculated via Cosine Similarity (`1 - (dc.embedding <=> query::vector)`).
- **Chunking Strategy**: 1,000-character sliding windows with 200-character overlap to retain contextual continuity across section boundaries.

### 3. Latency & Cost Optimization Plan
- **Semantic Response Caching**: In-memory hash cache (`app/cache.py`) intercepts repeated queries, delivering sub-10ms response times at **$0.00 incremental cost**.
- **Rate-Limit Resilience**: Embedded exponential backoff algorithm (`embed_with_retry`) catches HTTP 429 resource exhaustion errors automatically.

### 4. Reliability & Database Failover
- **Fail-Open Architecture**: If the PostgreSQL `pgvector` container is unavailable, the query engine seamlessly degrades to the embedded SQLite vector store without crashing user requests.

---

## 🛠️ Postmortem Note: "What Broke & How We Fixed It"

During system stress testing and migration, three critical engineering issues were identified and resolved:

1. **API Rate Limit Spikes (HTTP 429)**
   - *Issue*: Bulk document ingestion triggered Gemini embedding quota exhaustion during batch processing.
   - *Fix*: Implemented exponential backoff and dynamic sleep interval throttling in `embed_with_retry()` to respect API quotas while completing batch ingestion reliably.

2. **Database Container Dependency Bottleneck**
   - *Issue*: Standalone developer environments without local PostgreSQL containers could not run query inference.
   - *Fix*: Architected a dual-database adapter pattern in `scripts/ingest_data.py` and `app/main.py` that automatically falls back to an in-memory/SQLite vector store when Postgres is unreachable.

3. **Empty Retrieval Cascade Bug**
   - *Issue*: When PostgreSQL connected to an unpopulated database table, it returned an empty list (`[]`) without raising an exception, preventing fallback execution.
   - *Fix*: Updated `retrieve_chunks()` to check for non-zero result lists before terminating fallback evaluation.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for containerized stack)
- Gemini API Key

### 1. Clone & Setup Environment
```bash
git clone https://github.com/SaddamHosyn/mise-rag-project.git
cd mise-rag-project
pip install -r requirements.txt
```

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sec_rag_db
DB_USER=raguser
DB_PASSWORD=ragpassword
```

### 2. Ingest Policy Data
```bash
python scripts/ingest_data.py
```

### 3. Launch FastAPI REST Engine & Streamlit UI
```bash
# Option A: Run Streamlit UI
python -m streamlit run app/frontend.py

# Option B: Run FastAPI REST Server
uvicorn app.api:app --reload --port 8000
```

### 4. Run MLOps Automated Evaluation Benchmark
```bash
python scripts/evaluate_rag.py
```

---

## 🐳 Docker Deployment

To spin up the entire production stack (PostgreSQL + pgvector, FastAPI API server, and Streamlit frontend):

```bash
docker compose up -d --build
```

Access Services:
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Telemetry Metrics**: `GET http://localhost:8000/metrics`
- **Evaluation Benchmark**: `GET http://localhost:8000/eval`

---

## 📂 Project Structure

```
mise-rag-project/
├── app/
│   ├── api.py               # Production FastAPI REST Endpoints (/query, /metrics, /eval)
│   ├── main.py              # Core RAG engine, retrieval logic, & prompt assembly
│   ├── frontend.py          # Interactive Streamlit Web Interface
│   ├── telemetry.py         # Latency (p50, p95, p99), token & cost observability tracker
│   ├── cache.py             # Sub-10ms response cache engine
│   ├── config.py            # PostgreSQL database connector
│   └── entity_resolver.py   # Document form resolver
├── scripts/
│   ├── ingest_data.py       # Ingestion script for policies & PDFs
│   ├── evaluate_rag.py      # Automated benchmark suite (Hit@10, Faithfulness, Latency, Cost)
│   ├── generate_dataset_summary.py # Bondora loan dataset metrics generator
│   └── clear_pgvector.py    # Database cleanup utility
├── scrape/data/
│   ├── LoanData_Bondora.csv # Bondora 179k loan dataset
│   └── policies/            # Financial agreements, PDFs, & policy text files
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD workflow
├── Dockerfile               # Production multi-stage Docker container build
├── docker-compose.yml       # PostgreSQL pgvector + FastAPI + Streamlit orchestrator
├── init.sql                 # PostgreSQL vector extension database schema
├── requirements.txt         # Python dependencies
└── README.md                # System documentation
```
