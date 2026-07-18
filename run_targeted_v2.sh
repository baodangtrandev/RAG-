#!/bin/bash
set -e

# Nhận tham số truyền vào (ví dụ --limit 2)
LIMIT_CMD="$@"

export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"

# Tạo thư mục results_v2
mkdir -p results_v2

# Copy các baseline và eval tương ứng để tránh chạy lại tốn thời gian
echo "Sao chép các baseline có sẵn sang results_v2..."
cp results/baseline_bm25.jsonl results_v2/ 2>/dev/null || true
cp results/baseline_vector.jsonl results_v2/ 2>/dev/null || true
cp results/baseline_vector_reranker.jsonl results_v2/ 2>/dev/null || true

cp results/eval_baseline_bm25.json results_v2/ 2>/dev/null || true
cp results/eval_baseline_vector.json results_v2/ 2>/dev/null || true
cp results/eval_baseline_vector_reranker.json results_v2/ 2>/dev/null || true

echo "=========================================="
echo "CHẠY CÁC KỊCH BẢN TARGETED BENCHMARK (V2 - HYBRID SEARCH)"
echo "=========================================="

echo "[1/5] High-Recall (Tau=0.05, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results_v2/trag_targeted_high_recall.jsonl

echo "[2/5] Balanced (Tau=0.15, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v2/trag_targeted_balanced.jsonl

echo "[3/5] High-Speed (Tau=0.30, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results_v2/trag_targeted_high_speed.jsonl

echo "[4/5] Baseline Gamma (Tau=0.15, Gamma=1.0)"
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v2/trag_targeted_gamma_1.0.jsonl

echo "[5/5] T-RAG (No Reranker) (Tau=0.15, Gamma=0.0)"
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 --no-reranker -o results_v2/trag_targeted_no_reranker.jsonl

echo "=========================================="
echo "HOÀN TẤT TOÀN BỘ TARGETED BENCHMARK V2!"
echo "Kết quả thô và file đánh giá nằm ở: results_v2/"
echo "=========================================="
