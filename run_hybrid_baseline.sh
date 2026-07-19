#!/bin/bash
set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0

# Nhận tham số giới hạn (ví dụ --limit 2)
LIMIT_CMD="$@"

echo "=========================================="
echo "BƯỚC 1: CHẠY HYBRID SEARCH + RERANKER BASELINE"
echo "=========================================="
/network-volume/miniconda3/envs/trag/bin/python src/baselines/hybrid_search/run_hybrid.py $LIMIT_CMD -o results_v4/baseline_hybrid.jsonl

echo "=========================================="
echo "BƯỚC 2: KHỞI ĐỘNG SERVER VLLM ĐỂ CHẤM ĐIỂM BASELINE"
echo "=========================================="
# Khởi động server vLLM chấm điểm
/network-volume/miniconda3/envs/trag/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_LLM_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --max-model-len 8192 \
    --port 8000 > vllm_judge_hybrid.log 2>&1 &

VLLM_PID=$!

echo "Đang đợi server vLLM khởi động (khoảng 30 giây)..."
timeout 120 bash -c 'until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done' || {
    echo "Lỗi: Không thể kết nối tới vLLM server sau 120 giây. Xem vllm_judge_hybrid.log để biết chi tiết."
    kill $VLLM_PID
    exit 1
}

echo "=========================================="
echo "BƯỚC 3: CHẤM ĐIỂM BẰNG LLM JUDGE"
echo "=========================================="
/network-volume/miniconda3/envs/trag/bin/python -m src.scripts.metrics_based_eval \
    --answers-file results_v4/baseline_hybrid.jsonl \
    --results-file results_v4/eval_baseline_hybrid.json \
    --parallelism 16

echo "=========================================="
echo "BƯỚC 4: DỌN DẸP SERVER"
echo "=========================================="
kill $VLLM_PID
echo "Đã tắt server vLLM."

echo "Đang tạo báo cáo tổng hợp mới..."
/network-volume/miniconda3/envs/trag/bin/python generate_report.py results_v4
