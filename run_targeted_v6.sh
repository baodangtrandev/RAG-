#!/bin/bash
# run_targeted_v6.sh
# Benchmark 8 cấu hình mục tiêu (Targeted Combos Config A-H) đề xuất từ kết quả v6.2
# Đợt chạy này tập trung tìm ra điểm ngọt (Sweet Spot) tốt nhất bằng cách phối hợp các tham số tối ưu.

set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"

# Nhận tham số truyền vào (ví dụ --limit 2 hoặc --limit 500)
LIMIT_CMD="$@"

# Tạo thư mục lưu kết quả
OUT="results_targeted_v6"
mkdir -p $OUT

PYTHON="/network-volume/miniconda3/envs/trag/bin/python"
BENCH_V2="src/trag_v2/run_benchmark_v2.py"

echo "======================================================================"
# In ra thông tin cấu hình chạy để tiện kiểm tra
if [ -n "$LIMIT_CMD" ]; then
    echo "BẮT ĐẦU CHẠY TARGETED BENCHMARK VỚI THIẾT LẬP: $LIMIT_CMD"
else
    echo "BẮT ĐẦU CHẠY FULL TARGETED BENCHMARK (500 CÂU HỎI)"
fi
echo "======================================================================"

echo "[1/8] Running Config A: Ultimate Combo..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.0 --tau-alpha 0.15 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_targeted_combo_a.jsonl

echo "[2/8] Running Config B: Precision Strike..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.0 --tau-alpha 0.15 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_targeted_combo_b.jsonl

echo "[3/8] Running Config C: Gamma Zero Low Tau..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.0 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_targeted_combo_c.jsonl

echo "[4/8] Running Config D: Alpha Sweet Spot..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.4 --tau-alpha 0.15 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_targeted_combo_d.jsonl

echo "[5/8] Running Config E: Low Gamma High Alpha..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.0 --tau-alpha 0.12 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_targeted_combo_e.jsonl

echo "[6/8] Running Config F: Wide Net Balanced..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.0 --tau-alpha 0.08 --top-k-retrieve 25 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_targeted_combo_f.jsonl

echo "[7/8] Running Config G: Minimalist Best..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.4 --tau-alpha 0.15 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_targeted_combo_g.jsonl

echo "[8/8] Running Config H: Speed King v2..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.0 --tau-alpha 0.00 --top-k-final 5 \
    --smart-hop2 --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_targeted_combo_h.jsonl

echo "======================================================================"
echo "PHASE 2: VERIFYING / STARTING LOCAL JUDGE SERVER (VLLM)"
echo "======================================================================"

VLLM_PID=""
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "Phát hiện vLLM Judge server đã khởi chạy sẵn trên cổng 8000. Sử dụng server hiện tại."
else
    echo "Khởi động vLLM Judge server trên cổng 8000..."
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$JUDGE_LLM_MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
        --max-model-len 8192 \
        --port 8000 > vllm_judge_targeted.log 2>&1 &
    VLLM_PID=$!
    
    echo "Đang chờ server vLLM khởi động (khoảng 30 giây)..."
    timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
        echo "LỖI: vLLM server không khởi động được sau 120 giây. Xem chi tiết tại vllm_judge_targeted.log."
        kill $VLLM_PID || true
        exit 1
    }
    echo "Server vLLM Judge đã sẵn sàng!"
fi

echo "======================================================================"
echo "PHASE 3: EVALUATING TARGETED GENERATED ANSWERS SEQUENTIALLY"
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

if [ -n "$VLLM_PID" ]; then
    kill $VLLM_PID || true
    echo "Đã tắt server vLLM."
fi

echo "Đang tạo báo cáo tổng hợp kết quả ($OUT)..."
$PYTHON generate_report_v6.py $OUT
echo "Toàn bộ tiến trình run_targeted_v6 hoàn tất!"
