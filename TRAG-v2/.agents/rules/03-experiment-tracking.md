# Rule 03: Experiment Tracking & Reproducibility

## Scope
Áp dụng cho tất cả experiment trong Phase 4 (và bất kỳ thử nghiệm nào tạo ra kết quả số liệu).

---

## 1. Nguyên tắc cốt lõi

> **"Nếu không reproduce được thì không tồn tại."**
>
> Mọi số liệu trong bài báo PHẢI có thể tái tạo bằng một lệnh duy nhất.

---

## 2. Mọi experiment phải có config file riêng

### Cấu trúc `configs/exp/`

```
configs/
├── base.yaml               # Default shared config
├── retriever.yaml          # Retriever hyperparams
├── reranker.yaml           # Reranker hyperparams
└── exp/
    ├── exp-00-baseline.yaml
    ├── exp-01-hybrid-only.yaml
    ├── exp-02-query-expand.yaml
    ├── exp-03-time-decay.yaml
    ├── exp-04-metadata-filter.yaml
    └── exp-05-full-trag.yaml
```

### Ví dụ config file

```yaml
# configs/exp/exp-05-full-trag.yaml
experiment_id: "EXP-05"
description: "T-RAG Full System — all components enabled"

components:
  metadata_filter: true
  temporal_decay: true
  hybrid_search: true
  query_expansion: true

retriever:
  dense_model: "BAAI/bge-large-en-v1.5"
  top_k_dense: 50
  top_k_sparse: 50
  rrf_k: 60           # RRF constant

reranker:
  model: "BAAI/bge-reranker-v2-m3"
  top_k_final: 10
  lambda_decay: 0.01  # Time decay coefficient

llm:
  model: "meta-llama/Meta-Llama-3-8B-Instruct"
  engine: "vllm"
  max_new_tokens: 512
  temperature: 0.0    # Deterministic generation

seed: 42
batch_size: 32
```

---

## 3. Cấu trúc output experiment

Sau mỗi lần chạy experiment, output phải được lưu theo cấu trúc này:

```
results/
├── experiment_log.csv          # Master log của tất cả experiments
├── exp-00-baseline/
│   ├── metrics.json            # Số liệu tổng hợp
│   ├── per_query_results.jsonl # Kết quả từng query
│   └── run_info.json           # Metadata của run
├── exp-05-full-trag/
│   ├── metrics.json
│   ├── per_query_results.jsonl
│   └── run_info.json
└── figures/
    ├── main_results_bar.png
    └── ablation_heatmap.png
```

### Format `run_info.json`

```json
{
  "experiment_id": "EXP-05",
  "timestamp": "2026-09-01T14:30:00+07:00",
  "git_commit": "a3f8c21",
  "gpu": "H100-80GB",
  "cuda_version": "12.3",
  "python_version": "3.10.14",
  "seed": 42,
  "dataset": "EnterpriseRAG-Bench",
  "num_queries": 1000,
  "duration_seconds": 3620,
  "config_file": "configs/exp/exp-05-full-trag.yaml"
}
```

### Format `metrics.json`

```json
{
  "experiment_id": "EXP-05",
  "retrieval": {
    "mrr_at_10": 0.512,
    "recall_at_5": 0.643,
    "ndcg_at_10": 0.489,
    "recall_at_10": 0.721
  },
  "generation": {
    "answer_relevance": 0.78,
    "faithfulness": 0.82
  },
  "latency": {
    "retrieval_p50_ms": 420,
    "retrieval_p95_ms": 980,
    "e2e_p50_ms": 2100
  }
}
```

---

## 4. `experiment_log.csv` — Master tracker

| exp_id | timestamp | mrr@10 | recall@5 | ndcg@10 | p50_ms | git_commit | notes |
|--------|-----------|--------|---------|---------|--------|-----------|-------|
| EXP-00 | 2026-08-15 | 0.367 | 0.481 | 0.342 | 780 | abc1234 | Baseline |
| EXP-01 | 2026-08-16 | 0.412 | 0.538 | 0.389 | 820 | def5678 | +Hybrid |
| EXP-05 | 2026-09-01 | 0.512 | 0.643 | 0.489 | 420 | a3f8c21 | Full T-RAG |

---

## 5. Script chạy experiment chuẩn

```bash
# Chạy một experiment
python scripts/run_experiment.py --config configs/exp/exp-05-full-trag.yaml

# Chạy toàn bộ ablation study
bash scripts/run_all_experiments.sh

# Tổng hợp kết quả
python scripts/summarize_results.py --output results/experiment_log.csv
```

---

## 6. MLflow / W&B integration (khuyến nghị)

```python
import mlflow

with mlflow.start_run(run_name=cfg.experiment_id):
    mlflow.log_params(cfg.to_dict())
    
    # ... chạy experiment ...
    
    mlflow.log_metrics({
        "mrr_at_10": metrics["mrr_at_10"],
        "recall_at_5": metrics["recall_at_5"],
    })
    mlflow.log_artifact("results/exp-05-full-trag/metrics.json")
```

---

## ✅ Checklist mỗi experiment

- [ ] Có config YAML riêng trong `configs/exp/`
- [ ] `run_info.json` ghi đủ: git commit, GPU, seed, timestamp
- [ ] `metrics.json` ghi đủ tất cả metrics cần báo cáo
- [ ] `experiment_log.csv` đã được cập nhật
- [ ] Có thể reproduce bằng lệnh: `python scripts/run_experiment.py --config configs/exp/exp-XX.yaml`
