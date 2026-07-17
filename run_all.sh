#!/bin/bash
set -e

# ==============================================================================
# T-RAG Benchmark & Ablation Study Runner
# ==============================================================================
# Script này sẽ tự động hóa chạy một loạt các cấu hình nhằm phục vụ cho 
# Ablation Study trong Paper của bạn.
# Lưu ý: Yêu cầu chạy trong môi trường conda (conda activate trag) 
# và có đủ bộ nhớ VRAM.
# ==============================================================================

# Xoá kết quả cũ (nếu có) để đảm bảo run_all chạy sạch sẽ từ đầu
rm -rf results/
mkdir -p results
# Bỏ giới hạn (--limit) để chạy trên toàn bộ tập test
LIMIT_CMD="" 
# Nếu muốn test nhanh, đổi thành LIMIT_CMD="--limit 50"

echo "=========================================="
echo "PHẦN 1: BASELINES (Hệ quy chiếu)"
echo "=========================================="
python src/baselines/bm25/run_bm25.py $LIMIT_CMD -o results/baseline_bm25.jsonl
sleep 10
python src/baselines/vector_search/run_vector.py $LIMIT_CMD -o results/baseline_vector.jsonl
sleep 10
python src/baselines/vector_reranker/run_vector_reranker.py $LIMIT_CMD -o results/baseline_vector_reranker.jsonl
sleep 10

echo "=========================================="
echo "PHẦN 2: T-RAG - ABLATION TRÊN CSEP"
echo "=========================================="
# 1. Default (Có CSEP)
python src/run_benchmark.py $LIMIT_CMD --csep -o results/trag_csep_true.jsonl
sleep 10
# 2. Không CSEP
python src/run_benchmark.py $LIMIT_CMD --no-csep -o results/trag_csep_false.jsonl
sleep 10

echo "=========================================="
echo "PHẦN 3: T-RAG - ABLATION TRÊN NGƯỠNG TAU (ROUTER)"
echo "=========================================="
# Dựa theo báo cáo evaluation_report.txt, ta chạy các ngưỡng Tau từ 0.10 đến 0.50
python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results/trag_tau_0.05.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.10 -o results/trag_tau_0.10.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results/trag_tau_0.15.jsonl # Default
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.20 -o results/trag_tau_0.20.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.25 -o results/trag_tau_0.25.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results/trag_tau_0.30.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.40 -o results/trag_tau_0.40.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --tau 0.50 -o results/trag_tau_0.50.jsonl
sleep 10

echo "=========================================="
echo "PHẦN 4: T-RAG - ABLATION TRÊN RERANKER THRESHOLD"
echo "=========================================="
# Thay đổi biến môi trường RERANKER_THRESHOLD trực tiếp trong lúc chạy
# Ngưỡng 0.0 (Tắt filtering, chỉ rerank)
export RERANKER_THRESHOLD="0.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_rerank_thresh_0.0.jsonl
sleep 10

# Ngưỡng 1.0 (Loại bỏ docs nhiễu nhẹ)
export RERANKER_THRESHOLD="1.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_rerank_thresh_1.0.jsonl
sleep 10

# Ngưỡng 3.0 (Cắt rất gắt, chỉ giữ lại docs có tính liên quan cực cao)
export RERANKER_THRESHOLD="3.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_rerank_thresh_3.0.jsonl
sleep 10

# Reset lại môi trường
export RERANKER_THRESHOLD="0.0"

echo "=========================================="
echo "PHẦN 5: T-RAG - ABLATION TRÊN TOP-K"
echo "=========================================="
python src/run_benchmark.py $LIMIT_CMD --top-k-retrieve 10 --top-k-final 3 -o results/trag_topk_10_3.jsonl
sleep 10
python src/run_benchmark.py $LIMIT_CMD --top-k-retrieve 20 --top-k-final 5 -o results/trag_topk_20_5.jsonl # Default
sleep 10
python src/run_benchmark.py $LIMIT_CMD --top-k-retrieve 40 --top-k-final 10 -o results/trag_topk_40_10.jsonl
sleep 10

echo "=========================================="
echo "PHẦN 6: T-RAG - ABLATION TRÊN SW-RRF (GAMMA)"
echo "=========================================="
# Thay đổi trọng số ưu tiên nguồn (Bayesian Prior)
# Gamma = 0.0: Trở về thuật toán RRF truyền thống (Không ưu tiên nguồn)
export RAG_GAMMA="0.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_gamma_0.0.jsonl
sleep 10

# Gamma = 1.0: Ưu tiên tuyến tính
export RAG_GAMMA="1.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_gamma_1.0.jsonl
sleep 10

# Gamma = 2.0: Ưu tiên cấp số mũ (Default)
export RAG_GAMMA="2.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_gamma_2.0.jsonl
sleep 10

# Gamma = 3.0: Rất gắt, gần như chỉ tin tưởng các nguồn Top 1 của Router
export RAG_GAMMA="3.0"
python src/run_benchmark.py $LIMIT_CMD -o results/trag_gamma_3.0.jsonl
sleep 10

# Reset
export RAG_GAMMA="2.0"

echo "=========================================="
echo "🎉 Đã chạy xong TOÀN BỘ kịch bản tạo câu trả lời."
echo "=========================================="

echo "=========================================="
echo "PHẦN 7: TỰ ĐỘNG ĐÁNH GIÁ VỚI LLM JUDGE"
echo "=========================================="
# Đọc file .env để lấy JUDGE_LLM_MODEL
source .env

echo "[1/3] Khởi động vLLM API Server ($JUDGE_LLM_MODEL) ở background..."
# Tự động tắt vLLM Server nếu script bị thoát (Ctrl+C hoặc lỗi)
trap 'echo "Tắt vLLM Server (PID: $VLLM_PID)..."; kill $VLLM_PID 2>/dev/null' EXIT

python -m vllm.entrypoints.openai.api_server --model "$JUDGE_LLM_MODEL" --port 8000 --max-model-len 8192 &
VLLM_PID=$!

echo "[2/3] Đang đợi vLLM API Server sẵn sàng (có thể mất 1-3 phút để load model)..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    # Kiểm tra xem tiến trình vLLM còn sống không, tránh treo vòng lặp vô hạn nếu OOM
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "\n[LỖI] vLLM Server đã bị crash trong lúc khởi động (Có thể do OOM). Huỷ bỏ quá trình chấm điểm!"
        exit 1
    fi
    sleep 10
    echo -n "."
done
echo -e "\nvLLM API Server đã online!"

echo "[3/3] Tiến hành chấm điểm tất cả các file trong thư mục results/"
for file in results/*.jsonl; do
    # Bỏ qua nếu có file nào không phải là file chứa câu trả lời
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    filename=$(basename "$file")
    echo "--------------------------------------------------"
    echo "Đang chấm điểm: $filename"
    
    # Chạy script chấm điểm (metrics_based_eval.py)
    # Bổ sung --parallelism 16 để tận dụng sức mạnh batching của vLLM API
    python -m src.scripts.metrics_based_eval \
        --answers-file "$file" \
        --results-file "results/eval_${filename%.jsonl}.json" \
        --parallelism 16
done

echo "=========================================="
echo "🏆 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH BENCHMARK VÀ CHẤM ĐIỂM!"
echo "Các file báo cáo điểm số nằm ở: results/eval_*.json"
echo "=========================================="
