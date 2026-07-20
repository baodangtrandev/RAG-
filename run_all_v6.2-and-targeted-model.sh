#!/bin/bash
# run_all_v6.2.sh
# Toàn bộ quy trình chạy benchmark 53 cấu hình và đánh giá tự động (Baselines, T-RAG v1, T-RAG v2)
# ĐẢM BẢO TÍNH CÔNG BẰNG: Tất cả các kịch bản chạy với mức k_final = 5 (trừ opt_topk3 chạy với k_final = 3, opt_topk1 chạy với k_final = 1).
set -e

# Đọc cấu hình từ .env
source .env
export TORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export RAG_DB_URI="/tmp/lancedb"
export RAYON_NUM_THREADS=192
export OMP_NUM_THREADS=192
export RAG_HYBRID_SEARCH="True"

# OVERRIDE MODEL
export LOCAL_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export JUDGE_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"


# Nhận tham số truyền vào (ví dụ --limit 5 hoặc --limit 500)
LIMIT_CMD="$@"

# Tạo thư mục lưu kết quả
mkdir -p results6.2-model

PYTHON="/network-volume/miniconda3/envs/trag/bin/python"
BENCH_V1="src/run_benchmark.py"
BENCH_V2="src/trag_v2/run_benchmark_v2.py"
OUT="results6.2-model"

echo "======================================================================"
echo "PHASE 1: RUNNING 7 BASELINES (K_FINAL = 5)"
echo "======================================================================"
echo "[1/7] BM25 Baseline..."
$PYTHON src/baselines/bm25/run_bm25.py $LIMIT_CMD -o $OUT/baseline_bm25.jsonl

echo "[2/7] VECTOR Baseline..."
$PYTHON src/baselines/vector_search/run_vector.py $LIMIT_CMD -o $OUT/baseline_vector.jsonl

echo "[3/7] VECTOR_RERANKER Baseline..."
$PYTHON src/baselines/vector_reranker/run_vector_reranker.py $LIMIT_CMD -o $OUT/baseline_vector_reranker.jsonl

echo "[4/7] HYBRID Baseline..."
$PYTHON src/baselines/hybrid_search/run_hybrid.py $LIMIT_CMD -o $OUT/baseline_hybrid.jsonl

echo "[5/7] HyDE Baseline..."
$PYTHON src/baselines/hyde/run_hyde.py $LIMIT_CMD -o $OUT/baseline_hyde.jsonl

echo "[6/7] Query Expansion Baseline..."
$PYTHON src/baselines/query_expansion/run_query_expansion.py $LIMIT_CMD -o $OUT/baseline_query_expansion.jsonl

echo "[7/7] LLM Router Baseline..."
$PYTHON src/baselines/llm_router/run_llm_router.py $LIMIT_CMD -o $OUT/baseline_llm_router.jsonl


echo "======================================================================"
echo "PHASE 2: RUNNING 7 T-RAG V1 CONFIGURATIONS (K_FINAL = 5)"
echo "======================================================================"
# Cấu hình v1 tự động đọc RAG_TOP_K_FINAL=5 từ .env hoặc môi trường.

echo "[1/7] T-RAG v1 Balanced (Tau=0.15, Gamma=0.0)..."
export RAG_GAMMA="0.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.15 -o $OUT/trag_v1_balanced.jsonl

echo "[2/7] T-RAG v1 Balanced G1 (Tau=0.15, Gamma=1.0)..."
export RAG_GAMMA="1.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.15 -o $OUT/trag_v1_balanced_g1.jsonl

echo "[3/7] T-RAG v1 High-Speed (Tau=0.30, Gamma=0.0)..."
export RAG_GAMMA="0.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.30 -o $OUT/trag_v1_high_speed.jsonl

echo "[4/7] T-RAG v1 High-Speed G1 (Tau=0.30, Gamma=1.0)..."
export RAG_GAMMA="1.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.30 -o $OUT/trag_v1_high_speed_g1.jsonl

echo "[5/7] T-RAG v1 High-Recall (Tau=0.05, Gamma=0.0)..."
export RAG_GAMMA="0.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.05 -o $OUT/trag_v1_high_recall.jsonl

echo "[6/7] T-RAG v1 High-Recall G1 (Tau=0.05, Gamma=1.0)..."
export RAG_GAMMA="1.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.05 -o $OUT/trag_v1_high_recall_g1.jsonl

echo "[7/7] T-RAG v1 No Reranker (Tau=0.15, Gamma=0.0)..."
export RAG_GAMMA="0.0"
$PYTHON $BENCH_V1 $LIMIT_CMD --tau 0.15 --no-reranker -o $OUT/trag_v1_no_reranker.jsonl


echo "======================================================================"
echo "PHASE 3: RUNNING T-RAG V2 (GRID SEARCH & ABLATIONS) - 24 CONFIGS (K_FINAL = 5)"
echo "======================================================================"

# Standard v2
echo "[1/24] T-RAG v2 Standard (tau_base=0.15, gamma=0.5, alpha=0.08)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_standard.jsonl

# Grid Tau Base
echo "[2/24] T-RAG v2 Grid Tau 0.05..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_tau_0.05.jsonl

echo "[3/24] T-RAG v2 Grid Tau 0.10..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_tau_0.10.jsonl

echo "[4/24] T-RAG v2 Grid Tau 0.20..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_tau_0.20.jsonl

echo "[5/24] T-RAG v2 Grid Tau 0.30..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.30 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_tau_0.30.jsonl

# Grid Gamma
echo "[6/24] T-RAG v2 Grid Gamma 0.0..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.0 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_gamma_0.0.jsonl

echo "[7/24] T-RAG v2 Grid Gamma 0.3..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.3 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_gamma_0.3.jsonl

echo "[8/24] T-RAG v2 Grid Gamma 0.7..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.7 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_gamma_0.7.jsonl

echo "[9/24] T-RAG v2 Grid Gamma 1.0..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 1.0 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_gamma_1.0.jsonl

# Grid Alpha
echo "[10/24] T-RAG v2 Grid Alpha 0.00 (Same as Ablation No Adaptive Tau)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.00 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.00.jsonl

echo "[11/24] T-RAG v2 Grid Alpha 0.04..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.04.jsonl

echo "[12/24] T-RAG v2 Grid Alpha 0.12..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.12 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.12.jsonl

echo "[13/24] T-RAG v2 Grid Alpha 0.15..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.15 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.15.jsonl

echo "[14/24] T-RAG v2 Grid Alpha 0.25..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.25 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.25.jsonl

echo "[15/24] T-RAG v2 Grid Alpha 0.50..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.50 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_grid_alpha_0.50.jsonl

# Hybrid weights search
echo "[16/24] T-RAG v2 Dense Search Only (Dense=1.0, Sparse=0.0)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 1.0 --sparse-weight 0.0 \
    -o $OUT/trag_v2_dense_only.jsonl

echo "[17/24] T-RAG v2 Sparse Search Only (Dense=0.0, Sparse=1.0)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.0 --sparse-weight 1.0 \
    -o $OUT/trag_v2_sparse_only.jsonl

echo "[18/24] T-RAG v2 Hybrid Dense Heavy (Dense=0.7, Sparse=0.3)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.7 --sparse-weight 0.3 \
    -o $OUT/trag_v2_hybrid_dense_0.7.jsonl

echo "[19/24] T-RAG v2 Hybrid Sparse Heavy (Dense=0.3, Sparse=0.7)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/trag_v2_hybrid_dense_0.3.jsonl

echo "[20/24] T-RAG v2 Hybrid Dense Super-Heavy (Dense=0.9, Sparse=0.1)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.9 --sparse-weight 0.1 \
    -o $OUT/trag_v2_hybrid_dense_0.9.jsonl

echo "[21/24] T-RAG v2 Hybrid Sparse Super-Heavy (Dense=0.1, Sparse=0.9)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/trag_v2_hybrid_dense_0.1.jsonl

# Ablations
echo "[22/24] T-RAG v2 Ablation No Smart Hop 2..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --no-smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_ablation_no_smart_hop2.jsonl

echo "[23/24] T-RAG v2 Ablation No Adaptive Tau..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --no-adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_ablation_no_adaptive_tau.jsonl

echo "[24/24] T-RAG v2 Ablation No CSEP..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --no-csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/trag_v2_ablation_no_csep.jsonl


echo "======================================================================"
echo "PHASE 4: RUNNING 15 T-RAG V2 OPTIMIZED CONFIGS (K_FINAL = 5, EXCEPT TOPK3, TOPK1)"
echo "======================================================================"

echo "[1/15] OPT: Best Correctness (Alpha=0.25, D=0.1/S=0.9)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.25 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/opt_best_correctness.jsonl

echo "[2/15] OPT: Best Completeness (Alpha=0.04, D=0.3/S=0.7)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_best_completeness.jsonl

echo "[3/15] OPT: Low Latency (Tau=0.20, Gamma=0.3, Alpha=0.00)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.3 --tau-alpha 0.00 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_low_latency.jsonl

echo "[4/15] OPT: Balanced (Tau=0.15, Gamma=0.5, Alpha=0.04)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.04 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_balanced.jsonl

echo "[5/15] OPT: Middle Sparse (D=0.2, S=0.8)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.2 --sparse-weight 0.8 \
    -o $OUT/opt_d02_s08.jsonl

echo "[6/15] OPT: Middle Balanced (D=0.4, S=0.6)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.4 --sparse-weight 0.6 \
    -o $OUT/opt_d04_s06.jsonl

echo "[7/15] OPT: Gamma=0.4..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.4 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_gamma_04.jsonl

echo "[8/15] OPT: Max Perf (Tau=0.10, Alpha=0.04, D=0.1/S=0.9)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.10 --gamma 0.5 --tau-alpha 0.04 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/opt_max_performance.jsonl

echo "[9/15] OPT: Speed+Sparse (Tau=0.20, Alpha=0.00, D=0.3/S=0.7)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.20 --gamma 0.5 --tau-alpha 0.00 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_speed_sparse.jsonl

# opt_topk3: Thay thế opt_topk5 (do topk5 trùng với v2 standard ở mức k=5).
# Cấu hình này giúp đánh giá độ nhạy khi chỉ lấy Top 3 docs (tiết kiệm tối đa context).
echo "[10/15] OPT: TopK=3 (Standard, but K_final=3)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 3 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_topk3.jsonl

echo "[11/15] OPT: TopK Retrieve = 10 (Standard, retrieve depth 10)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-retrieve 10 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_topk_retrieve_10.jsonl

echo "[12/15] OPT: TopK Retrieve = 30 (Standard, retrieve depth 30)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-retrieve 30 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_topk_retrieve_30.jsonl

echo "[13/15] OPT: TopK Final = 1 (Standard, but K_final=1)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.15 --gamma 0.5 --tau-alpha 0.08 --top-k-final 1 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.5 --sparse-weight 0.5 \
    -o $OUT/opt_topk1.jsonl

echo "[14/15] OPT: High Recall + Sparse Heavy (Tau=0.05, D=0.3/S=0.7)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.05 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.3 --sparse-weight 0.7 \
    -o $OUT/opt_high_recall_sparse_heavy.jsonl

echo "[15/15] OPT: High Speed + Sparse Super-Heavy (Tau=0.30, D=0.1/S=0.9)..."
$PYTHON $BENCH_V2 $LIMIT_CMD \
    --tau-base 0.30 --gamma 0.5 --tau-alpha 0.08 --top-k-final 5 \
    --smart-hop2 --adaptive-tau --csep --dense-weight 0.1 --sparse-weight 0.9 \
    -o $OUT/opt_high_speed_sparse_super_heavy.jsonl



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
