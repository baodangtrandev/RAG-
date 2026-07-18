#!/bin/bash
set -e

# Nhận tham số truyền vào (ví dụ --limit 2)
LIMIT_CMD="$@"

export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"
export RAG_DENSE_WEIGHT="0.3"
export RAG_SPARSE_WEIGHT="0.7"

# Tạo thư mục results_v4
mkdir -p results_v4

echo "=========================================="
echo "CHẠY 3 BASELINES (V3)"
echo "=========================================="
/network-volume/miniconda3/envs/trag/bin/python src/baselines/bm25/run_bm25.py $LIMIT_CMD -o results_v4/baseline_bm25.jsonl
/network-volume/miniconda3/envs/trag/bin/python src/baselines/vector_search/run_vector.py $LIMIT_CMD -o results_v4/baseline_vector.jsonl
/network-volume/miniconda3/envs/trag/bin/python src/baselines/vector_reranker/run_vector_reranker.py $LIMIT_CMD -o results_v4/baseline_vector_reranker.jsonl

echo "=========================================="
echo "CHẠY CÁC KỊCH BẢN TARGETED BENCHMARK (V3 - HYBRID SEARCH 0.3/0.7)"
echo "=========================================="

echo "[1/7] High-Recall (Tau=0.05, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results_v4/trag_targeted_high_recall.jsonl

echo "[2/7] Balanced (Tau=0.15, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v4/trag_targeted_balanced.jsonl

echo "[3/7] High-Speed (Tau=0.30, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results_v4/trag_targeted_high_speed.jsonl

echo "[4/7] High-Recall Gamma 1.0 (Tau=0.05, Gamma=1.0)"
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results_v4/trag_targeted_high_recall_gamma_1.0.jsonl

echo "[5/7] Balanced Gamma 1.0 (Tau=0.15, Gamma=1.0)"
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v4/trag_targeted_gamma_1.0.jsonl

echo "[6/7] High-Speed Gamma 1.0 (Tau=0.30, Gamma=1.0)"
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results_v4/trag_targeted_high_speed_gamma_1.0.jsonl

echo "[7/7] T-RAG (No Reranker) (Tau=0.15, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 --no-reranker -o results_v4/trag_targeted_no_reranker.jsonl

echo "=========================================="
echo "HOÀN TẤT TOÀN BỘ TARGETED BENCHMARK V3!"
echo "Kết quả thô nằm ở: results_v4/"
echo "=========================================="
