# Workflow: Phase 2 — Data Setup & Baseline Evaluation

## Trigger
Phase 1 done (research proposal approved). Bắt đầu dựng môi trường và đo baselines.

## Goal
Môi trường chạy ổn định, 500k docs đã index vào LanceDB, baseline scores đã đo xong.

---

## STEP 2.1 — Setup môi trường

### 2.1a — Khởi tạo project structure

```bash
# Từ thư mục TRAG-v2/
mkdir -p src/{ingestion,retrieval,query_parser,reranker,generation}
mkdir -p tests/{unit,integration}
mkdir -p scripts configs/exp notebooks results/figures docs/lit_review

# Tạo __init__.py cho tất cả packages
find src/ -type d -exec touch {}/__init__.py \;
touch tests/__init__.py
```

### 2.1b — Tạo `environment.yml`

```yaml
# environment.yml
name: trag
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - lancedb>=0.6.0
    - sentence-transformers>=2.7.0
    - vllm>=0.4.0
    - rank-bm25>=0.2.2
    - pydantic>=2.0.0
    - typer>=0.12.0
    - mlflow>=2.12.0
    - ragas>=0.1.9
    - pytest>=8.0.0
    - ruff>=0.4.0
    - black>=24.3.0
    - mypy>=1.9.0
    - pre-commit>=3.7.0
    - nbstripout>=0.7.0
    - jupyter>=1.0.0
    - pandas>=2.2.0
    - matplotlib>=3.8.0
    - seaborn>=0.13.0
    - tqdm>=4.66.0
    - pyyaml>=6.0.0
```

```bash
conda env create -f environment.yml
conda activate trag
pre-commit install
nbstripout --install
```

### 2.1c — Tạo `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "trag"
version = "0.1.0"
description = "T-RAG: Temporal & Targeted RAG System"
requires-python = ">=3.10"

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.mypy]
strict = true
python_version = "3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## STEP 2.2 — Implement Data Ingestion Pipeline

### 2.2a — Thiết kế LanceDB schema

```python
# src/ingestion/lance_schema.py
import lancedb
from lancedb.pydantic import LanceModel, Vector

class DocumentSchema(LanceModel):
    doc_id: str
    content: str
    source_type: str          # slack, gmail, jira, confluence
    timestamp: str            # ISO 8601 format
    author: str | None = None
    thread_id: str | None = None
    vector: Vector(1024)      # bge-large-en-v1.5 dimension
```

### 2.2b — Chunking strategy (Semantic, không fixed-size)

```python
# src/ingestion/chunker.py
# Dùng sentence boundaries + overlap
# Không dùng fixed token count
```

### 2.2c — Chạy ingestion

```bash
# Ingest toàn bộ dataset
python scripts/ingest.py \
  --data-dir /path/to/EnterpriseRAG-Bench \
  --db-path ./data/lancedb \
  --batch-size 512 \
  --embedding-model BAAI/bge-large-en-v1.5
```

**Kiểm tra sau ingest:**
```python
import lancedb
db = lancedb.connect("./data/lancedb")
tbl = db.open_table("documents")
print(f"Total documents: {tbl.count_rows()}")  # Phải ≈ 500k
```

---

## STEP 2.3 — Implement Standard RAG Baseline

**File:** `scripts/baseline_rag.py`

**Logic:**
1. Embed query với `bge-large-en-v1.5`
2. ANN search trên toàn bộ LanceDB (top-50)
3. Concatenate top-10 docs làm context
4. Generate answer với Llama-3-8B

```bash
python scripts/baseline_rag.py \
  --config configs/exp/exp-00-baseline.yaml \
  --queries data/eval_queries.jsonl \
  --output results/exp-00-baseline/
```

---

## STEP 2.4 — Implement BM25-only Baseline

**File:** `scripts/baseline_bm25.py`

```bash
python scripts/baseline_bm25.py \
  --config configs/exp/exp-00b-bm25.yaml \
  --queries data/eval_queries.jsonl \
  --output results/exp-00b-bm25/
```

---

## STEP 2.5 — Chạy Evaluation

**Tool:** RAGAS + custom BEIR-style eval

```bash
# Tính metrics từ kết quả
python scripts/eval.py \
  --predictions results/exp-00-baseline/per_query_results.jsonl \
  --ground-truth data/eval_ground_truth.jsonl \
  --output results/exp-00-baseline/metrics.json

# Expected output format:
# {
#   "mrr_at_10": 0.367,
#   "recall_at_5": 0.481,
#   "ndcg_at_10": 0.342
# }
```

---

## STEP 2.6 — Error Analysis Notebook

**File:** `notebooks/01_baseline_error_analysis.ipynb`

**Phân tích cần làm:**
1. Nhóm queries theo loại: temporal / keyword-heavy / semantic
2. Tính performance theo từng nhóm
3. Lấy 20 query mà baseline fail nhiều nhất
4. Phân tích root cause: recall fail hay precision fail?

```python
# Trong notebook
import pandas as pd

df = pd.read_json("results/exp-00-baseline/per_query_results.jsonl", lines=True)
df_temporal = df[df["query_type"] == "temporal"]
print(f"Baseline MRR@10 on temporal queries: {df_temporal['mrr_at_10'].mean():.3f}")
```

---

## ✅ Phase 2 Done Criteria

- [ ] `conda activate trag` hoạt động
- [ ] `pytest tests/ -v` — tất cả tests pass
- [ ] `tbl.count_rows()` ≈ 500k documents đã được index
- [ ] `results/exp-00-baseline/metrics.json` — có đủ MRR, Recall, NDCG
- [ ] `results/exp-00b-bm25/metrics.json` — BM25 baseline score
- [ ] Notebook error analysis chỉ ra được specific failure modes

## Estimated Time: 2 tuần
