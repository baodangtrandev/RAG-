# Workflow: Phase 3 — T-RAG Core Implementation

## Trigger
Phase 2 done. Baselines đã chạy, biết cụ thể Standard RAG fail ở đâu.

## Goal
Implement đầy đủ 4 module của T-RAG với unit tests, chạy được end-to-end.

---

## Sprint 3A — Storage Layer (Week 5)

### STEP 3.1 — Implement `lance_schema.py`

**Branch:** `feat/phase3-storage-layer`

**Checklist:**
- [ ] Schema có đủ: `doc_id`, `content`, `source_type`, `timestamp`, `vector`
- [ ] Index Dense: IVF_PQ hoặc HNSW
- [ ] Index FTS (Tantivy): enabled cho field `content`
- [ ] Metadata fields có thể filter được bằng SQL-style query

```python
# src/ingestion/lance_schema.py
import lancedb
from lancedb.pydantic import LanceModel, Vector
from datetime import datetime

class DocumentSchema(LanceModel):
    """Unified document schema for LanceDB.
    
    Supports: Dense search, FTS (Tantivy), and SQL metadata filtering.
    """
    doc_id: str
    content: str
    source_type: str       # "slack" | "gmail" | "jira" | "confluence"
    timestamp: datetime
    author: str = ""
    thread_id: str = ""
    vector: Vector(1024)   # BAAI/bge-large-en-v1.5 embedding
```

### STEP 3.2 — Implement `chunker.py`

**Chiến lược chunking:** Semantic chunking (KHÔNG dùng fixed token count)

**Logic:**
1. Split theo câu (sentence tokenizer)
2. Merge câu liên tiếp nếu cùng chủ đề (embedding similarity > threshold)
3. Overlap 1-2 câu giữa các chunk

```bash
# Test chunker
pytest tests/unit/test_chunker.py -v
```

**Test cases bắt buộc:**
- [ ] Chunk không bị cut mid-sentence
- [ ] Overlap đúng số câu config
- [ ] Metadata được propagate đúng vào mỗi chunk

### STEP 3.3 — Unit test ingestion

```bash
pytest tests/unit/test_ingestion.py -v
# Phải pass: test_schema_creation, test_chunker, test_metadata_extraction
```

---

## Sprint 3B — Query Parser (Week 6)

### STEP 3.4 — Implement `temporal_detector.py`

**Branch:** `feat/phase3-query-parser`

**Logic:** Detect nếu query cần thông tin mới nhất:

```python
# src/query_parser/temporal_detector.py

TEMPORAL_KEYWORDS = [
    "latest", "recent", "current", "newest", "now",
    "today", "this week", "this month", "update",
    "mới nhất", "gần đây", "hiện tại",
]

def detect_temporal_intent(query: str) -> bool:
    """Return True if query requires latest information."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in TEMPORAL_KEYWORDS)
```

**Lưu ý:** Với LLM-based detection (chính xác hơn):
```
Prompt: "Does this query require the most recent information? 
         Query: {query}
         Answer with JSON: {"requires_latest": true/false, "reason": "..."}"
```

### STEP 3.5 — Implement `metadata_classifier.py`

**Input:** User query
**Output:** `source_filter: str | None` — "slack", "gmail", "jira", "confluence", hoặc None (search all)

```
Prompt: "Which document source would most likely contain the answer?
         Sources: slack (chat messages), gmail (emails), jira (tickets), confluence (docs)
         Query: {query}
         Answer: source_type or 'all'"
```

### STEP 3.6 — Implement `expander.py` (HyDE + Query Expansion)

**HyDE strategy:**
```
Prompt: "Write a hypothetical document that would answer this question:
         Question: {query}
         Hypothetical Document:"
```

**Query expansion:**
```
Prompt: "Generate 3 alternative phrasings of this search query:
         Query: {query}
         Alternatives (JSON array):"
```

### STEP 3.7 — Test Query Parser

```bash
pytest tests/unit/test_query_parser.py -v
```

**Test cases bắt buộc:**
- [ ] Temporal query → `requires_latest=True`
- [ ] Historical query → `requires_latest=False`  
- [ ] Slack-specific query → `source_filter="slack"`
- [ ] General query → `source_filter=None`
- [ ] HyDE tạo document có độ dài hợp lý (50-200 words)

---

## Sprint 3C — Hybrid Retriever & Reranker (Week 7)

### STEP 3.8 — Implement `rrf_fusion.py`

**Branch:** `feat/phase3-hybrid-retriever`

**Công thức RRF:**
```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge dense and sparse ranked lists using RRF.
    
    RRF Score = Σ 1 / (k + rank_i)
    """
    scores: dict[str, float] = {}
    
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Unit test:**
```bash
pytest tests/unit/test_rrf_fusion.py -v
# Test: known ranking → expected RRF score
```

### STEP 3.9 — Implement `temporal_reranker.py`

**Công thức Time Decay:**

```
Final_Score = Relevance × e^(−λ × Δt)

Trong đó:
- Relevance: Cross-encoder score (0 → 1)
- λ: decay coefficient (từ config, default=0.01)
- Δt: số ngày kể từ ngày tài liệu (datetime.now() - doc.timestamp).days
- Nếu requires_latest=False → λ = 0 (không áp dụng decay)
```

```python
import math
from datetime import datetime

def compute_temporal_score(
    relevance: float,
    doc_timestamp: datetime,
    lambda_decay: float,
    requires_latest: bool,
) -> float:
    if not requires_latest:
        return relevance
    delta_t = (datetime.now() - doc_timestamp).days
    return relevance * math.exp(-lambda_decay * delta_t)
```

### STEP 3.10 — Integration Test

```bash
pytest tests/integration/test_pipeline_e2e.py -v
```

**Test case:**
```python
def test_e2e_pipeline_on_sample_queries():
    pipeline = TRAGPipeline.from_config("configs/exp/exp-05-full-trag.yaml")
    sample_queries = load_jsonl("data/sample_queries_100.jsonl")
    
    for query in sample_queries:
        result = pipeline.run(query["text"])
        assert result.answer is not None
        assert len(result.retrieved_docs) > 0
        assert result.latency_ms < 5000  # < 5 seconds per query
```

---

## STEP 3.11 — Hoàn thiện `pipeline.py`

```python
# src/pipeline.py
class TRAGPipeline:
    """Orchestrates the full T-RAG pipeline."""
    
    def run(self, query: str) -> PipelineOutput:
        # Step 1: Query Parser
        ctx = self.query_parser.parse(query)
        
        # Step 2: Hybrid Retrieval (top-50 candidates)
        candidates = self.retriever.retrieve(
            query=ctx.expanded_queries,
            source_filter=ctx.source_filter,
            top_k=self.cfg.retriever.top_k_dense,
        )
        
        # Step 3: Temporal Reranking (top-50 → top-10)
        reranked = self.reranker.rerank(
            candidates=candidates,
            requires_latest=ctx.requires_latest,
        )
        
        # Step 4: Generate answer
        answer = self.generator.generate(
            query=query,
            context=reranked[:self.cfg.reranker.top_k_final],
        )
        
        return PipelineOutput(answer=answer, retrieved_docs=reranked)
```

---

## ✅ Phase 3 Done Criteria

- [ ] `pytest tests/ -v` — 100% pass
- [ ] `ruff check src/` — 0 lỗi
- [ ] `mypy src/` — 0 type error
- [ ] `python scripts/run_pipeline.py --query "What is the latest deploy status?"` chạy end-to-end
- [ ] Mọi hyperparameter nằm trong YAML config (không hardcode)
- [ ] Latency trên 100 sample queries < 5 giây/query

## Estimated Time: 3 tuần
