#!/bin/bash
set -e

echo "=========================================="
echo "BẮT ĐẦU CHẤM ĐIỂM CÁC FILE ĐÃ HOÀN THNH"
echo "=========================================="

for file in results/*.jsonl; do
    filename=$(basename "$file")
    eval_file="results/eval_${filename%.jsonl}.json"
    
    if [ ! -f "$eval_file" ]; then
        echo "Đang chấm điểm: $filename"
        python -m src.scripts.metrics_based_eval \
            --answers-file "$file" \
            --results-file "$eval_file" \
            --parallelism 16
    else
        echo "Đã có điểm cho $filename, bỏ qua."
    fi
done

echo "=========================================="
echo "HOÀN THÀNH CHẤM ĐIỂM!"
echo "=========================================="
