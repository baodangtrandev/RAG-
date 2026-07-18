#!/bin/bash
set -e
echo "=========================================="
echo "KHỞI CHẠY SMOKE TEST CHO TARGETED BENCHMARK V3"
echo "=========================================="
./run_targeted_v3.sh --limit 2
echo "=========================================="
echo "SMOKE TEST TARGETED BENCHMARK V3 HOÀN TẤT!"
echo "=========================================="
