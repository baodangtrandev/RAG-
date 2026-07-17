#!/bin/bash
set -e

# ==============================================================================
# T-RAG Smoke Tests Runner
# ==============================================================================
# Script này dùng để test nhanh toàn bộ pipeline (BM25, Vector, Reranker, T-RAG)
# với số lượng câu hỏi cực nhỏ (5 câu) để đảm bảo không còn lỗi (Babbling, FTS)
# trước khi chạy benchmark thật.
# ==============================================================================

# Tạo thư mục chứa kết quả smoke test
mkdir -p smoke_results

LIMIT_CMD="--limit 5"

echo "=========================================="
echo "SMOKE TEST: BASELINES"
echo "=========================================="
echo "[1/4] Chạy BM25 Baseline (Kiểm tra lỗi FTS)..."
python src/baselines/bm25/run_bm25.py $LIMIT_CMD -o smoke_results/baseline_bm25.jsonl
sleep 10

echo "[2/4] Chạy Vector Search Baseline (Kiểm tra per-query latency)..."
python src/baselines/vector_search/run_vector.py $LIMIT_CMD -o smoke_results/baseline_vector.jsonl
sleep 10

echo "[3/4] Chạy Vector + Reranker Baseline..."
python src/baselines/vector_reranker/run_vector_reranker.py $LIMIT_CMD -o smoke_results/baseline_vector_reranker.jsonl
sleep 10

echo "=========================================="
echo "SMOKE TEST: T-RAG PIPELINE"
echo "=========================================="
echo "[4/4] Chạy T-RAG với CSEP (Kiểm tra lỗi Entity Babbling và Stop tokens)..."
python src/run_benchmark.py $LIMIT_CMD --csep -o smoke_results/trag_csep_true.jsonl
sleep 10

echo "=========================================="
echo "SMOKE TEST: EVALUATION (LLM JUDGE)"
echo "=========================================="
source .env

echo "[1/3] Khởi động vLLM API Server ($JUDGE_LLM_MODEL) ở background..."
trap 'echo "Tắt vLLM Server (PID: $VLLM_PID)..."; kill $VLLM_PID 2>/dev/null' EXIT

python -m vllm.entrypoints.openai.api_server --model "$JUDGE_LLM_MODEL" --port 8000 --max-model-len 8192 &
VLLM_PID=$!

echo "[2/3] Đang đợi vLLM API Server sẵn sàng (có thể mất 1-3 phút để load model)..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "\n[LỖI] vLLM Server đã bị crash trong lúc khởi động (Có thể do OOM). Huỷ bỏ quá trình chấm điểm!"
        exit 1
    fi
    sleep 10
    echo -n "."
done
echo -e "\nvLLM API Server đã online!"

echo "[3/3] Tiến hành chấm điểm tất cả các file trong thư mục smoke_results/"
for file in smoke_results/*.jsonl; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    filename=$(basename "$file")
    echo "--------------------------------------------------"
    echo "Đang chấm điểm: $filename"
    
    python -m src.scripts.metrics_based_eval \
        --answers-file "$file" \
        --results-file "smoke_results/eval_${filename%.jsonl}.json" \
        --parallelism 16
done

echo "=========================================="
echo "✅ HOÀN TẤT SMOKE TEST TOÀN DIỆN!"
echo "Kết quả được lưu tại thư mục: smoke_results/"
echo "Bạn có thể kiểm tra:"
echo "  - CSEP entity extraction đã sạch chưa (file trag_csep_true.jsonl)."
echo "  - BM25 đã trả về tài liệu thay vì báo 'I do not have enough information' chưa."
echo "  - Các câu trả lời đã hết bị lặp từ (babbling) chưa."
echo "  - Latency per-query đã được ghi nhận chưa."
echo "  - Eval script có chạy thành công không (các file eval_*.json)."
echo "=========================================="
