#!/bin/bash
# test_smoke_v6.2.sh
# Chạy smoke test nhanh (2 câu hỏi) cho bộ 45 kịch bản của run_all_v6.2.sh
set -e

echo "Khởi chạy smoke test v6.2..."
bash run_all_v6.2.sh --limit 2
echo "Smoke test v6.2 hoàn thành!"
