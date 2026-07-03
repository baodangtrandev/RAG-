# Workflow: Phase 4 — Ablation Study & Benchmarking

## Trigger
Phase 3 done. T-RAG full pipeline chạy ổn định.

## Goal
Đo hiệu suất đầy đủ theo ablation matrix 6 experiments, có đủ số liệu để viết paper.

---

## STEP 4.1 — Chuẩn bị Evaluation Set

### Xác định Eval Split

```python
# scripts/prepare_eval_set.py
# Load evaluation queries từ EnterpriseRAG-Bench
# Annotate query type: temporal / keyword-heavy / semantic / mixed

# Output: data/eval_queries_annotated.jsonl
# Format mỗi dòng:
{
  "query_id": "q001",
  "query": "What is the latest deployment status of service X?",
  "query_type": "temporal",           # temporal | keyword | semantic | mixed
  "relevant_doc_ids": ["doc_123", "doc_456"],
  "ground_truth_answer": "..."
}
```

### Subsets cần tạo

| Subset | Số lượng | Mục đích |
|--------|---------|---------|
| Full eval set | ~1000 queries | Đo metric chính |
| Temporal subset | ~200 queries | Chứng minh Time Decay hiệu quả |
| Keyword subset | ~200 queries | Chứng minh BM25 cần thiết |
| Source-specific subset | ~200 queries | Chứng minh Metadata Filtering |

---

## STEP 4.2 — Chạy 6 Experiments

### Ablation Matrix

| Config | Metadata Filter | Time Decay | Hybrid Search | Query Expansion |
|--------|:-:|:-:|:-:|:-:|
| `exp-00-baseline.yaml` | ❌ | ❌ | ❌ | ❌ |
| `exp-01-hybrid-only.yaml` | ❌ | ❌ | ✅ | ❌ |
| `exp-02-query-expand.yaml` | ❌ | ❌ | ❌ | ✅ |
| `exp-03-time-decay.yaml` | ❌ | ✅ | ❌ | ❌ |
| `exp-04-metadata-filter.yaml` | ✅ | ❌ | ❌ | ❌ |
| `exp-05-full-trag.yaml` | ✅ | ✅ | ✅ | ✅ |

### Script chạy toàn bộ

```bash
#!/bin/bash
# scripts/run_all_experiments.sh

set -e  # Dừng nếu có lỗi

EXPERIMENTS=(
  "configs/exp/exp-00-baseline.yaml"
  "configs/exp/exp-01-hybrid-only.yaml"
  "configs/exp/exp-02-query-expand.yaml"
  "configs/exp/exp-03-time-decay.yaml"
  "configs/exp/exp-04-metadata-filter.yaml"
  "configs/exp/exp-05-full-trag.yaml"
)

for cfg in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Running: $cfg"
  echo "========================================"
  python scripts/run_experiment.py --config "$cfg"
  echo "Done: $cfg"
  echo ""
done

# Tổng hợp kết quả
python scripts/summarize_results.py --output results/experiment_log.csv
echo "All experiments complete! Results in results/experiment_log.csv"
```

```bash
# Chạy
chmod +x scripts/run_all_experiments.sh
nohup bash scripts/run_all_experiments.sh > logs/ablation_run.log 2>&1 &
tail -f logs/ablation_run.log
```

---

## STEP 4.3 — Phân tích Kết quả

### Notebook: `notebooks/02_ablation_analysis.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load experiment log
df = pd.read_csv("results/experiment_log.csv")

# --- Plot 1: Main results bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ["mrr_at_10", "recall_at_5", "ndcg_at_10"]
labels = ["MRR@10", "Recall@5", "NDCG@10"]

for ax, metric, label in zip(axes, metrics, labels):
    bars = ax.bar(df["exp_id"], df[metric], color=["gray"]*5 + ["#2196F3"])
    bars[-1].set_color("#F44336")  # T-RAG in red
    ax.set_title(label)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("paper/figures/ablation_bar.pdf", bbox_inches="tight")

# --- Plot 2: Heatmap của component contribution ---
# Δ improvement over baseline for each component
```

### Phân tích theo loại query

```python
# So sánh T-RAG vs Baseline trên temporal queries
df_temporal = load_per_query_results("temporal")
print(f"Baseline MRR@10 (temporal): {df_temporal['exp00_mrr'].mean():.3f}")
print(f"T-RAG MRR@10 (temporal):    {df_temporal['exp05_mrr'].mean():.3f}")
print(f"Improvement: {(df_temporal['exp05_mrr'] - df_temporal['exp00_mrr']).mean()*100:.1f}%")
```

---

## STEP 4.4 — Case Study Analysis

### Notebook: `notebooks/03_case_study.ipynb`

**Chọn 5 query T-RAG thắng (T-RAG tốt hơn baseline nhiều nhất):**

```python
df_diff = df_per_query.copy()
df_diff["improvement"] = df_diff["exp05_mrr"] - df_diff["exp00_mrr"]

# Queries T-RAG thắng nhiều nhất
top_wins = df_diff.nlargest(5, "improvement")
# Queries T-RAG thua (honest limitation)
top_losses = df_diff.nsmallest(5, "improvement")
```

**Template case study (dùng trong paper):**

```markdown
### Case: Temporal Query Win
Query: "What is the current on-call schedule for the infra team?"
- Standard RAG retrieved doc from 2022 (high semantic similarity)
- T-RAG with Time Decay: doc from 2024 ranked #1
- Result: Correct answer vs. outdated information

### Case: Failure Case  
Query: "What did John say about the API in the meeting?"
- T-RAG misclassified as non-temporal → missed recent Slack message
- Root cause: Vague temporal signal in query
```

---

## STEP 4.5 — Latency Benchmark

```bash
# Đo latency trên 100 queries
python scripts/latency_benchmark.py \
  --config configs/exp/exp-05-full-trag.yaml \
  --n-queries 100 \
  --output results/latency_benchmark.json
```

**Output cần có:**

```json
{
  "retrieval_p50_ms": 420,
  "retrieval_p95_ms": 980,
  "reranking_p50_ms": 280,
  "e2e_p50_ms": 2100,
  "e2e_p95_ms": 4800,
  "throughput_qps": 0.48
}
```

---

## STEP 4.6 — Cập nhật Table 1 draft với số liệu thật

```markdown
# Table 1 — Cập nhật với số liệu thật

| System              | MRR@10 | Recall@5 | NDCG@10 |
|---------------------|--------|---------|---------|
| BM25-only           | X.XXX  | X.XXX   | X.XXX   |
| Dense-only          | X.XXX  | X.XXX   | X.XXX   |
| Standard RAG        | X.XXX  | X.XXX   | X.XXX   |
| T-RAG (Ours)        |**X.XXX**|**X.XXX**|**X.XXX**|
```

---

## ✅ Phase 4 Done Criteria

- [ ] Cả 6 experiments chạy xong, không có lỗi
- [ ] `results/experiment_log.csv` có đủ 6 rows với đủ metrics
- [ ] T-RAG (EXP-05) **tốt hơn baseline** trên MRR@10 và Recall@5
- [ ] Ablation study chứng minh contribution của từng component
- [ ] Có ít nhất 3 case study (win + loss)
- [ ] Latency benchmark hoàn chỉnh
- [ ] Figures đã được export dưới dạng PDF vector
- [ ] Table 1 đã có số liệu thật

## Estimated Time: 2 tuần
