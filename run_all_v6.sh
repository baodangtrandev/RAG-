#!/bin/bash
# run_all_v6.sh
# Toàn bộ quy trình chạy benchmark 35 cấu hình và đánh giá tự động (T-RAG v2, v1 và Baselines)
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

# Nhận tham số truyền vào (ví dụ --limit 5 hoặc --limit 500)
LIMIT_CMD="$@"

# Tạo thư mục lưu kết quả
mkdir -p results_v6

echo "======================================================================"
echo "PHASE 1: RUNNING 4 BASELINES"
echo "======================================================================"
echo "[1/4] BM25 Baseline..."
/network-volume/miniconda3/envs/trag/bin/python src/baselines/bm25/run_bm25.py $LIMIT_CMD -o results_v6/baseline_bm25.jsonl

echo "[2/4] VECTOR Baseline..."
/network-volume/miniconda3/envs/trag/bin/python src/baselines/vector_search/run_vector.py $LIMIT_CMD -o results_v6/baseline_vector.jsonl

echo "[3/4] VECTOR_RERANKER Baseline..."
/network-volume/miniconda3/envs/trag/bin/python src/baselines/vector_reranker/run_vector_reranker.py $LIMIT_CMD -o results_v6/baseline_vector_reranker.jsonl

echo "[4/4] HYBRID Baseline..."
/network-volume/miniconda3/envs/trag/bin/python src/baselines/hybrid_search/run_hybrid.py $LIMIT_CMD -o results_v6/baseline_hybrid.jsonl


echo "======================================================================"
echo "PHASE 2: RUNNING 7 T-RAG V1 CONFIGURATIONS"
echo "======================================================================"

echo "[1/7] T-RAG v1 Balanced (Tau=0.15, Gamma=0.0)..."
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v6/trag_v1_balanced.jsonl

echo "[2/7] T-RAG v1 Balanced G1 (Tau=0.15, Gamma=1.0)..."
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results_v6/trag_v1_balanced_g1.jsonl

echo "[3/7] T-RAG v1 High-Speed (Tau=0.30, Gamma=0.0)..."
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results_v6/trag_v1_high_speed.jsonl

echo "[4/7] T-RAG v1 High-Speed G1 (Tau=0.30, Gamma=1.0)..."
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results_v6/trag_v1_high_speed_g1.jsonl

echo "[5/7] T-RAG v1 High-Recall (Tau=0.05, Gamma=0.0)..."
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results_v6/trag_v1_high_recall.jsonl

echo "[6/7] T-RAG v1 High-Recall G1 (Tau=0.05, Gamma=1.0)..."
export RAG_GAMMA="1.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results_v6/trag_v1_high_recall_g1.jsonl

echo "[7/7] T-RAG v1 No Reranker (Tau=0.15, Gamma=0.0)..."
export RAG_GAMMA="0.0"
/network-volume/miniconda3/envs/trag/bin/python src/run_benchmark.py $LIMIT_CMD --tau 0.15 --no-reranker -o results_v6/trag_v1_no_reranker.jsonl


echo "======================================================================"
echo "PHASE 3: RUNNING T-RAG V2 (STANDARD, GRID SEARCH & ABLATIONS) - 24 CONFIGS"
echo "======================================================================"

# Standard
echo "[1/24] T-RAG v2 Standard (tau_base=0.15, gamma=0.5, alpha=0.08)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_standard.jsonl

# Grid Tau Base
echo "[2/24] T-RAG v2 Grid Tau 0.05..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_tau_0.05.jsonl

echo "[3/24] T-RAG v2 Grid Tau 0.10..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_tau_0.10.jsonl

echo "[4/24] T-RAG v2 Grid Tau 0.20..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_tau_0.20.jsonl

echo "[5/24] T-RAG v2 Grid Tau 0.30..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.30 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_tau_0.30.jsonl

# Grid Gamma
echo "[6/24] T-RAG v2 Grid Gamma 0.0..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.0 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_gamma_0.0.jsonl

echo "[7/24] T-RAG v2 Grid Gamma 0.3..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.3 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_gamma_0.3.jsonl

echo "[8/24] T-RAG v2 Grid Gamma 0.7..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.7 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_gamma_0.7.jsonl

echo "[9/24] T-RAG v2 Grid Gamma 1.0..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 1.0 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_gamma_1.0.jsonl

# Grid Alpha
echo "[10/24] T-RAG v2 Grid Alpha 0.00 (Same as Ablation No Adaptive Tau)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.00 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.00.jsonl

echo "[11/24] T-RAG v2 Grid Alpha 0.04..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.04.jsonl

echo "[12/24] T-RAG v2 Grid Alpha 0.12..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.12 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.12.jsonl

echo "[13/24] T-RAG v2 Grid Alpha 0.15..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.15 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.15.jsonl

echo "[14/24] T-RAG v2 Grid Alpha 0.25..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.25 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.25.jsonl

echo "[15/24] T-RAG v2 Grid Alpha 0.50..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.50 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_grid_alpha_0.50.jsonl

# Hybrid weights search
echo "[16/24] T-RAG v2 Dense Search Only (Dense=1.0, Sparse=0.0)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 1.0 --sparse-weight 0.0 \
    -o results_v6/trag_v2_dense_only.jsonl

echo "[17/24] T-RAG v2 Sparse Search Only (Dense=0.0, Sparse=1.0)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.0 --sparse-weight 1.0 \
    -o results_v6/trag_v2_sparse_only.jsonl

echo "[18/24] T-RAG v2 Hybrid Dense Heavy (Dense=0.7, Sparse=0.3)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.7 --sparse-weight 0.3 \
    -o results_v6/trag_v2_hybrid_dense_0.7.jsonl

echo "[19/24] T-RAG v2 Hybrid Sparse Heavy (Dense=0.3, Sparse=0.7)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o results_v6/trag_v2_hybrid_dense_0.3.jsonl

echo "[20/24] T-RAG v2 Hybrid Dense Super-Heavy (Dense=0.9, Sparse=0.1)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.9 --sparse-weight 0.1 \
    -o results_v6/trag_v2_hybrid_dense_0.9.jsonl

echo "[21/24] T-RAG v2 Hybrid Sparse Super-Heavy (Dense=0.1, Sparse=0.9)..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o results_v6/trag_v2_hybrid_dense_0.1.jsonl

# Ablations
echo "[22/24] T-RAG v2 Ablation No Smart Hop 2..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --no-smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_ablation_no_smart_hop2.jsonl

echo "[23/24] T-RAG v2 Ablation No Adaptive Tau..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --no-adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_ablation_no_adaptive_tau.jsonl

echo "[24/24] T-RAG v2 Ablation No CSEP..."
/network-volume/miniconda3/envs/trag/bin/python src/trag_v2/run_benchmark_v2.py $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --no-csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o results_v6/trag_v2_ablation_no_csep.jsonl


echo "======================================================================"
echo "PHASE 4: STARTING LOCAL JUDGE SERVER (VLLM)"
echo "======================================================================"
/network-volume/miniconda3/envs/trag/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_LLM_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --max-model-len 8192 \
    --port 8000 > vllm_judge_v6.log 2>&1 &

VLLM_PID=$!

echo "Đang chờ server vLLM khởi động (khoảng 30 giây)..."
timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
    echo "LỖI: vLLM server không khởi động được sau 120s. Xem vllm_judge_v6.log."
    kill $VLLM_PID
    exit 1
}
echo "Server vLLM Judge đã sẵn sàng!"


echo "======================================================================"
echo "PHASE 5: EVALUATING GENERATED ANSWERS SEQUENTIALLY"
echo "======================================================================"
for file in results_v6/*.jsonl; do
    filename=$(basename "$file")
    eval_file="results_v6/eval_${filename%.jsonl}.json"
    
    echo "Đang chấm điểm: $filename..."
    /network-volume/miniconda3/envs/trag/bin/python -m src.scripts.metrics_based_eval \
        --answers-file "$file" \
        --results-file "$eval_file" \
        --parallelism 16
done


echo "======================================================================"
echo "PHASE 6: CLEANING UP & GENERATING COMPARATIVE REPORT"
echo "======================================================================"
kill $VLLM_PID
echo "Đã tắt server vLLM."

echo "Đang tạo báo cáo tổng hợp kết quả (results_v6)..."
/network-volume/miniconda3/envs/trag/bin/python generate_report_v6.py results_v6
echo "Toàn bộ tiến trình run_all_v6 hoàn tất!"
