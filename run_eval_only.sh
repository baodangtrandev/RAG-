#!/bin/bash
set -e

# Đọc cấu hình từ .env
source .env

echo "=========================================="
echo "BƯỚC 1: KHỞI ĐỘNG SERVER VLLM ĐỂ CHẤM ĐIỂM"
echo "=========================================="
echo "Model: $JUDGE_LLM_MODEL"

# Chạy server vllm trong background
python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_LLM_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --port 8000 > vllm_server.log 2>&1 &

VLLM_PID=$!

echo "Đang đợi server vLLM khởi động (khoảng 30 giây)..."
# Chờ cho đến khi server mở cổng 8000
timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
    echo "Lỗi: Không thể kết nối tới vLLM server sau 120 giây. Xem vllm_server.log để biết chi tiết."
    kill $VLLM_PID
    exit 1
}

echo "Server vLLM đã sẵn sàng!"

echo "=========================================="
echo "BƯỚC 2: CHẤM ĐIỂM CÁC FILE JSONL"
echo "=========================================="

for file in results/*.jsonl; do
    filename=$(basename "$file")
    eval_file="results/eval_${filename%.jsonl}.json"
    
    echo "------------------------------------------"
    if [ ! -f "$eval_file" ] || grep -q '"average_correctness_pct": 0.0' "$eval_file"; then
        echo "Đang chấm điểm: $filename"
        python -m src.scripts.metrics_based_eval \
            --answers-file "$file" \
            --results-file "$eval_file" \
            --parallelism 16
    else
        echo "Đã có điểm chuẩn cho $filename, bỏ qua."
    fi
done

echo "=========================================="
echo "BƯỚC 3: DỌN DẸP & TỔNG HỢP"
echo "=========================================="
kill $VLLM_PID
echo "Đã tắt server vLLM."

echo "Đang tạo báo cáo tổng hợp..."
python generate_report.py
