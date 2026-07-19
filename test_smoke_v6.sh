#!/bin/bash
set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"
export RAG_DENSE_WEIGHT="0.3"
export RAG_SPARSE_WEIGHT="0.7"

mkdir -p results_v6

echo "Testing HYBRID Baseline..."
/network-volume/miniconda3/envs/trag/bin/python src/baselines/hybrid_search/run_hybrid.py --limit 2 -o results_v6/smoke_baseline_hybrid.jsonl

echo "Testing T-RAG v1 Balanced..."
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py --limit 2 --tau 0.15 -o results_v6/smoke_trag_v1_balanced.jsonl

echo "Testing T-RAG v2 Standard (with CSEP)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py --limit 2 \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/smoke_trag_v2_standard.jsonl

echo "Testing T-RAG v2 (NO CSEP)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py --limit 2 \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --no-csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/smoke_trag_v2_no_csep.jsonl

echo "Smoke test v6 completed successfully!"
