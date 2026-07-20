#!/bin/bash
# eval_with_qwen_judge.sh
# Dùng để đánh giá lại toàn bộ câu trả lời của Mistral-7B bằng Qwen-14B làm Judge.

set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192

# Cố định Judge model là Qwen-14B
export JUDGE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

SRC_DIR="results6.2-model"
OUT_DIR="results6.2-model-qwen_judge"
PYTHON="/network-volume/miniconda3/envs/trag/bin/python"

# Tạo thư mục đầu ra
mkdir -p "$OUT_DIR"

echo "======================================================================"
# Copy toàn bộ file câu trả lời (.jsonl) của Mistral-7B sang folder mới
echo "Copying answer files (.jsonl) to $OUT_DIR..."
# Sử dụng cp thay vì loop để tối ưu
cp "$SRC_DIR"/*.jsonl "$OUT_DIR/"
echo "Copy completed."

echo "======================================================================"
echo "PHASE 1: STARTING LOCAL JUDGE SERVER (VLLM QWEN-14B)"
echo "======================================================================"

VLLM_PID=""
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "Phát hiện vLLM server đang chạy sẵn trên cổng 8000. Cần kiểm tra xem có đúng Qwen-14B không."
    echo "Nếu không đúng, vui lòng tắt vLLM cũ trước khi chạy script này."
else
    echo "Khởi động vLLM Judge server (Qwen-14B) trên cổng 8000..."
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$JUDGE_LLM_MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.80 \
        --max-model-len 8192 \
        --port 8000 > vllm_judge_qwen_re-eval.log 2>&1 &
    VLLM_PID=$!
    
    echo "Đang chờ server vLLM (Qwen-14B) khởi động (khoảng 30 giây)..."
    timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
        echo "LỖI: vLLM server không khởi động được sau 120 giây. Xem vllm_judge_qwen_re-eval.log."
        kill $VLLM_PID || true
        exit 1
    }
    echo "Server vLLM Judge (Qwen-14B) đã sẵn sàng!"
fi

echo "======================================================================"
echo "PHASE 2: EVALUATING GENERATED ANSWERS WITH QWEN-14B JUDGE"
echo "======================================================================"
for file in "$OUT_DIR"/*.jsonl; do
    filename=$(basename "$file")
    eval_file="$OUT_DIR/eval_${filename%.jsonl}.json"
    
    echo "Đang chấm điểm bằng Qwen-14B: $filename..."
    $PYTHON -m src.scripts.metrics_based_eval \
        --answers-file "$file" \
        --results-file "$eval_file" \
        --parallelism 16
done

echo "======================================================================"
echo "PHASE 3: CLEANING UP & GENERATING REPORT"
echo "======================================================================"

if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" || true
    echo "Đã tắt server vLLM Qwen-14B."
fi

echo "Đang tạo báo cáo tổng hợp kết quả tại $OUT_DIR..."
$PYTHON generate_report_v6.py "$OUT_DIR"
echo "Quá trình đánh giá lại bằng Qwen-14B hoàn tất!"
