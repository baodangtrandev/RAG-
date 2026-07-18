#!/bin/bash
set -e

# Không truyền limit để chạy toàn bộ tập dữ liệu (500 câu)
LIMIT_CMD=""

echo "=========================================="
echo "CHẠY CÁC KỊCH BẢN TARGETED BENCHMARK"
echo "=========================================="

echo "[1/4] High-Recall (Tau=0.05, Gamma=0.0)"
# Mục tiêu: Đánh bại Baseline về Điểm số. Quét toàn bộ 9 bảng, không phạt nguồn.
export RAG_GAMMA="0.0"
python src/run_benchmark.py $LIMIT_CMD --tau 0.05 -o results/trag_targeted_high_recall.jsonl


echo "[2/4] Balanced (Tau=0.15, Gamma=0.0)"
# Mục tiêu: Cân bằng giữa Tốc độ và Điểm số. Quét 4-7 bảng, không phạt nguồn.
export RAG_GAMMA="0.0"
python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results/trag_targeted_balanced.jsonl


echo "[3/4] High-Speed (Tau=0.30, Gamma=0.0)"
# Mục tiêu: Đánh bại Baseline về Latency. Chỉ quét 2-3 bảng chắc chắn nhất, không phạt nguồn.
export RAG_GAMMA="0.0"
python src/run_benchmark.py $LIMIT_CMD --tau 0.30 -o results/trag_targeted_high_speed.jsonl


echo "[4/4] Baseline Gamma (Tau=0.15, Gamma=1.0)"
# Mục tiêu: Kịch bản có phạt nguồn nhưng nhẹ hơn (Gamma=1.0) để so sánh với Gamma=0.0 ở trên
export RAG_GAMMA="1.0"
python src/run_benchmark.py $LIMIT_CMD --tau 0.15 -o results/trag_targeted_gamma_1.0.jsonl


echo "=========================================="
echo "HOÀN TẤT TOÀN BỘ TARGETED BENCHMARK!"
echo "Các file báo cáo điểm số nằm ở: results/eval_trag_targeted_*.json"
echo "=========================================="
