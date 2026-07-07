# Workflow: Phase 2 — Data Setup & Baseline Evaluation

## Trigger
Phase 1 done (research proposal approved). Bắt đầu dựng môi trường và đo baselines.

## Goal
Môi trường chạy ổn định, 511k docs đã được sharding thành nhiều bảng (tables) trong LanceDB theo `source_type`, baseline scores đã đo xong.

---

## STEP 2.1 — Setup môi trường

### 2.1a — Khởi tạo project structure

```bash
# Từ thư mục TRAG-v2/
mkdir -p src/{ingestion,retrieval,routing,reranker,generation}
mkdir -p tests/{unit,integration}
mkdir -p scripts configs/exp notebooks results/figures
```

### 2.1b — Tạo `environment.yml`

```yaml
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
    - rank-bm25>=0.2.2
    - pydantic>=2.0.0
    - typer>=0.12.0
    - pytest>=8.0.0
    - pandas>=2.2.0
    - pyarrow>=15.0.0
    - scikit-learn>=1.4.0
    - tqdm>=4.66.0
```

### 2.1c — Tạo `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "trag"
version = "0.1.0"
description = "T-RAG: Targeted RAG System"
requires-python = ">=3.10"
```

---

## STEP 2.2 — Implement Data Ingestion Pipeline (Database Sharding)

### 2.2a — Thiết kế LanceDB Sharded Schema

Thay vì một bảng, chúng ta sẽ tạo $N$ bảng tương ứng với các `source_type`.

```python
# src/ingestion/lance_schema.py
import lancedb
from lancedb.pydantic import LanceModel, Vector

class DocumentSchema(LanceModel):
    doc_id: str
    content: str
    title: str | None = None
    source_type: str
    vector: Vector(1024)      # bge-large-en-v1.5 dimension
```

### 2.2b — Chạy ingestion & sharding

```bash
# Ingest toàn bộ dataset và shard theo source_type
python scripts/ingest.py \
  --data-dir data/EnterpriseRAG-Bench/data/documents \
  --db-path ./data/lancedb \
  --batch-size 512 \
  --embedding-model BAAI/bge-large-en-v1.5
```

**Kiểm tra sau ingest:**
```python
import lancedb
db = lancedb.connect("./data/lancedb")
print(db.table_names()) # Nên ra các tên như: slack, jira, github, gmail...
```

---

## STEP 2.3 — Implement Standard RAG Baseline

**Logic:** Tìm kiếm mù quáng trên TẤT CẢ các bảng, lấy top-100 rồi tổng hợp lại lấy top-10.
Ghi lại thời gian chạy (Latency) và số tài liệu phải scan (Search Space = 100%).

---

## STEP 2.4 — Implement Metadata Filtered RAG (Hard Filter)

**Logic:** Dùng nhãn `source_type` chuẩn để search đúng bảng duy nhất (như một upper bound của Router).
Ghi lại thời gian chạy và Recall.

---

## ✅ Phase 2 Done Criteria

- [ ] Project structure và môi trường Python đã sẵn sàng
- [ ] LanceDB chứa nhiều bảng phân mảnh theo nguồn
- [ ] Các metrics baseline (Standard vs. Filtered) được ghi nhận
