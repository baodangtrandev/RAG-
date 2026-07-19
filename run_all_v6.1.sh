#!/bin/bash
# run_all_v6.1.sh
# Benchmark bổ sung: 10 cấu hình "cherry-pick" tối ưu nhất của T-RAG v2
# Dựa trên phân tích kết quả v6 để kết hợp các tham số tốt nhất từ từng grid search
#
# Insights từ v6:
#   - Tau: 0.15 tốt nhất cho Correctness, 0.20 balance tốt (latency giảm, correctness gần bằng)
#   - Gamma: 0.5 tối ưu, 0.3 cũng rất tốt
#   - Alpha: 0.04 → Completeness cao nhất (46.9%), 0.25 → Correctness cao nhất (36.7%)
#   - Weights: D=0.1/S=0.9 → Correctness cao nhất (36.6%), D=0.3/S=0.7 → Completeness cao nhất (47.6%)
#   - Chưa test: gamma=0.4, D=0.2/S=0.8, D=0.4/S=0.6, combo params, top_k_final variations
set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"

# Nhận tham số truyền vào (ví dụ --limit 5 hoặc --limit 500)
LIMIT_CMD="$@"

# Tạo thư mục lưu kết quả
mkdir -p results_v6.1

PYTHON="/network-volume/miniconda3/envs/trag/bin/python"
BENCH="src/trag_v2/run_benchmark_v2.py"
OUT="results_v6.1"

echo "======================================================================"
echo "  T-RAG v2 OPTIMIZED CONFIGS BENCHMARK (v6.1)"
echo "  10 cấu hình cherry-pick tối ưu nhất"
echo "======================================================================"

# =====================================================================
# CONFIG 1: Best Correctness Combo
# Kết hợp alpha=0.25 (correctness winner) + D=0.1/S=0.9 (correctness winner)
# Dự đoán: Correctness > 36.7% nhờ synergy
# =====================================================================
echo "[1/10] Best Correctness Combo (Alpha=0.25, D=0.1/S=0.9)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.25 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/opt_best_correctness.jsonl

# =====================================================================
# CONFIG 2: Best Completeness Combo
# Kết hợp alpha=0.04 (completeness winner) + D=0.3/S=0.7 (completeness winner)
# Dự đoán: Completeness > 47.6%
# =====================================================================
echo "[2/10] Best Completeness Combo (Alpha=0.04, D=0.3/S=0.7)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_best_completeness.jsonl

# =====================================================================
# CONFIG 3: Low Latency Champion
# tau=0.20 (tốc độ cao) + gamma=0.3 + static alpha + D=0.5/S=0.5
# Dự đoán: Total Latency < 0.95s, Correctness ~36%
# =====================================================================
echo "[3/10] Low Latency Champion (Tau=0.20, Gamma=0.3, Alpha=0.00)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.3 --tau-alpha 0.00 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_low_latency.jsonl

# =====================================================================
# CONFIG 4: Balanced Optimal v2
# tau=0.15 (chuẩn) + gamma=0.5 + alpha=0.04 (completeness sweet spot)
# Dự đoán: Cân bằng tốt nhất giữa Correctness + Completeness
# =====================================================================
echo "[4/10] Balanced Optimal (Tau=0.15, Gamma=0.5, Alpha=0.04, D=0.5/S=0.5)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_balanced.jsonl

# =====================================================================
# CONFIG 5: Unexplored Middle Weights (D=0.2/S=0.8)
# Trọng số giữa D=0.1/S=0.9 và D=0.3/S=0.7 — chưa test trong v6
# Dự đoán: Có thể là sweet spot giữa Correctness và Completeness
# =====================================================================
echo "[5/10] Middle Sparse Weights (D=0.2, S=0.8)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.2 --sparse-weight 0.8 \
    -o $OUT/opt_d02_s08.jsonl

# =====================================================================
# CONFIG 6: Unexplored Middle Weights (D=0.4/S=0.6)
# Trọng số giữa D=0.3/S=0.7 và D=0.5/S=0.5 — chưa test trong v6
# Dự đoán: Có thể giữ Correctness cao hơn D=0.3/S=0.7
# =====================================================================
echo "[6/10] Middle Balanced Weights (D=0.4, S=0.6)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_d04_s06.jsonl

# =====================================================================
# CONFIG 7: Unexplored Gamma=0.4
# Gamma giữa 0.3 (35.7%) và 0.5 (36.3%) — chưa test trong v6
# Dự đoán: Có thể là sweet spot thực sự
# =====================================================================
echo "[7/10] Gamma=0.4 (Unexplored, D=0.5/S=0.5)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.4 --tau-alpha 0.08 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_gamma_04.jsonl

# =====================================================================
# CONFIG 8: Max Performance Combo
# tau=0.10 (recall cao) + gamma=0.5 + alpha=0.04 + D=0.1/S=0.9
# Combo các winner: tau thấp cho recall + sparse heavy cho correctness
# Dự đoán: Correctness cao nhất có thể, latency ~1.03s
# =====================================================================
echo "[8/10] Max Performance Combo (Tau=0.10, Alpha=0.04, D=0.1/S=0.9)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.5 --tau-alpha 0.04 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/opt_max_performance.jsonl

# =====================================================================
# CONFIG 9: Speed + Sparse Heavy
# tau=0.20 + alpha=0.00 (no overhead) + D=0.3/S=0.7 (completeness)
# Dự đoán: Latency thấp nhất khả dĩ (~0.91s) mà vẫn giữ Completeness cao
# =====================================================================
echo "[9/10] Speed + Sparse Heavy (Tau=0.20, Alpha=0.00, D=0.3/S=0.7)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.5 --tau-alpha 0.00 --top-k-final 7 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_speed_sparse.jsonl

# =====================================================================
# CONFIG 10: Top-K Final Variation (top_k=5)
# Cùng config Standard nhưng giảm context window xuống 5 docs
# Dự đoán: Latency giảm nhẹ, Correctness có thể tăng nếu bớt noise
# =====================================================================
echo "[10/10] Standard + TopK=5 (Tau=0.15, Gamma=0.5, Alpha=0.08, K=5)..."
$PYTHON $BENCH $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_topk5.jsonl


echo "======================================================================"
echo "PHASE 2: STARTING LOCAL JUDGE SERVER (VLLM)"
echo "======================================================================"
/network-volume/miniconda3/envs/trag/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_LLM_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --max-model-len 8192 \
    --port 8000 > vllm_judge_v6.1.log 2>&1 &

VLLM_PID=$!

echo "Đang chờ server vLLM khởi động (khoảng 30 giây)..."
timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
    echo "LỖI: vLLM server không khởi động được sau 120s. Xem vllm_judge_v6.1.log."
    kill $VLLM_PID
    exit 1
}
echo "Server vLLM Judge đã sẵn sàng!"


echo "======================================================================"
echo "PHASE 3: EVALUATING GENERATED ANSWERS SEQUENTIALLY"
echo "======================================================================"
for file in $OUT/*.jsonl; do
    filename=$(basename "$file")
    eval_file="$OUT/eval_${filename%.jsonl}.json"

    echo "Đang chấm điểm: $filename..."
    $PYTHON -m src.scripts.metrics_based_eval \
        --answers-file "$file" \
        --results-file "$eval_file" \
        --parallelism 16
done


echo "======================================================================"
echo "PHASE 4: CLEANING UP & GENERATING COMPARATIVE REPORT"
echo "======================================================================"
kill $VLLM_PID
echo "Đã tắt server vLLM."

echo "Đang tạo báo cáo tổng hợp kết quả ($OUT)..."
$PYTHON generate_report_v6.py $OUT
echo "Toàn bộ tiến trình run_all_v6.1 hoàn tất!"
