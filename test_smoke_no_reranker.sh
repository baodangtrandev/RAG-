#!/bin/bash
set -e
export PATH="/network-volume/miniconda3/envs/trag/bin:$PATH"
mkdir -p smoke_results
echo "[2/2] Smoke Test: No-Reranker"
export RAG_GAMMA="0.0"
python src/run_benchmark.py --limit 1 --tau 0.15 --no-reranker -o smoke_results/test_no_reranker.jsonl
echo "DONE!"
