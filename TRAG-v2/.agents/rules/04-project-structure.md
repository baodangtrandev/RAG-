# Rule 04: Project Structure & Module Design

## Scope
Quy định cấu trúc thư mục, interface giữa các module, và quy tắc dependency.

---

## 1. Cấu trúc thư mục chuẩn

```
TRAG-v2/
├── .agents/
│   ├── skills/              # Installed skills
│   ├── rules/               # Project rules (file này)
│   └── workflows/           # Automated workflows
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Pydantic config models
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py       # Semantic chunking
│   │   ├── metadata_extractor.py
│   │   └── lance_schema.py  # LanceDB table schema
│   ├── query_parser/
│   │   ├── __init__.py
│   │   ├── expander.py      # Query expansion via LLM
│   │   ├── metadata_classifier.py  # source_type detection
│   │   ├── temporal_detector.py    # requires_latest flag
│   │   └── hyde.py          # Hypothetical Document Embedding
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   └── rrf_fusion.py
│   ├── reranker/
│   │   ├── __init__.py
│   │   ├── temporal_reranker.py   # Cross-encoder + Time Decay
│   │   └── context_truncator.py
│   ├── generation/
│   │   ├── __init__.py
│   │   └── answer_generator.py    # vLLM interface
│   └── pipeline.py          # Orchestrate all modules
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_rrf_fusion.py
│   │   ├── test_temporal_reranker.py
│   │   └── test_query_parser.py
│   └── integration/
│       └── test_pipeline_e2e.py
│
├── scripts/
│   ├── ingest.py            # Ingest toàn bộ dataset vào LanceDB
│   ├── run_experiment.py    # Chạy 1 experiment từ config file
│   ├── run_all_experiments.sh  # Chạy toàn bộ ablation matrix
│   ├── eval.py              # Tính metrics từ results
│   └── summarize_results.py # Tổng hợp experiment_log.csv
│
├── configs/
│   ├── base.yaml
│   ├── retriever.yaml
│   ├── reranker.yaml
│   └── exp/
│       ├── exp-00-baseline.yaml
│       └── exp-05-full-trag.yaml
│
├── notebooks/
│   ├── 01_baseline_error_analysis.ipynb
│   ├── 02_ablation_analysis.ipynb
│   ├── 03_case_study.ipynb
│   └── 04_visualization.ipynb
│
├── results/
│   ├── experiment_log.csv
│   └── figures/
│
├── docs/
│   ├── architecture.md
│   └── api_reference.md
│
├── .pre-commit-config.yaml
├── pyproject.toml
├── environment.yml
├── Dockerfile
└── README.md
```

---

## 2. Interface giữa các module

Mỗi module phải export một **public interface** rõ ràng qua `__init__.py`.

### Data model chuẩn (`src/config.py`)

```python
from pydantic import BaseModel
from datetime import datetime

class Document(BaseModel):
    """Unified document model dùng xuyên suốt pipeline."""
    doc_id: str
    content: str
    source_type: str           # "slack", "gmail", "jira", "confluence"
    timestamp: datetime
    metadata: dict[str, str] = {}

class QueryContext(BaseModel):
    """Kết quả sau Query Parser."""
    original_query: str
    expanded_queries: list[str]
    source_filter: str | None   # None = search all
    requires_latest: bool       # Trigger time decay
    hyde_document: str | None   # Hypothetical document

class RetrievalResult(BaseModel):
    """Kết quả từ Hybrid Retriever."""
    document: Document
    rrf_score: float
    dense_score: float
    sparse_score: float

class RerankResult(BaseModel):
    """Kết quả sau Temporal Reranker."""
    document: Document
    final_score: float         # relevance × e^(-λΔt)
    relevance_score: float     # Cross-encoder score
    time_penalty: float        # e^(-λΔt)
```

---

## 3. Dependency Flow (chỉ một chiều)

```
pipeline.py
    ├── query_parser/   (không import module khác trong src/)
    ├── retrieval/      (chỉ import từ ingestion/)
    ├── reranker/       (chỉ nhận input từ retrieval/)
    └── generation/     (chỉ nhận input từ reranker/)
```

**Quy tắc:** Module cấp thấp hơn **không được** import module cấp cao hơn.
- ❌ `retrieval/` import từ `reranker/` → Vi phạm
- ✅ `reranker/` import từ `retrieval/` (nhận RetrievalResult) → OK

---

## 4. Quy tắc về notebooks

- Notebooks **chỉ được** import từ `src/` và `results/`.
- Notebooks **không** chứa business logic (phải extract vào `src/` nếu cần dùng lại).
- Notebooks phải có **numbered prefix** (01_, 02_, ...) để rõ thứ tự đọc.
- Notebooks phải được **cleared output** trước khi commit (dùng `nbstripout`).

```bash
pip install nbstripout
nbstripout --install  # Auto-strip khi git add
```

---

## 5. Quy tắc về `scripts/`

Scripts là entry point, không phải library. Chúng:
- **Nhận argument** từ CLI (`argparse` hoặc `typer`)
- **Import** từ `src/`
- **Log progress** rõ ràng
- **Return exit code** phù hợp (0 = success, 1 = error)

```python
# scripts/run_experiment.py
import typer
from src.pipeline import TRAGPipeline
from src.config import ExperimentConfig

app = typer.Typer()

@app.command()
def main(config: str = typer.Option(..., help="Path to experiment YAML config")):
    cfg = ExperimentConfig.from_yaml(config)
    pipeline = TRAGPipeline(cfg)
    pipeline.run_eval()

if __name__ == "__main__":
    app()
```
