#!/bin/bash
cd "$(dirname "$0")/.."

echo "Step 1: Running document ingestion & embedding pipeline..."
python scripts/ingest_data.py

echo "Step 2: Running automated evaluation benchmark..."
python scripts/evaluate_rag.py

echo "Pipeline execution complete!"
