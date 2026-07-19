# T-RAG v2 — Thiết kế tối ưu Performance & Latency

## Bối cảnh & Vấn đề

T-RAG v1 đạt kết quả "giữa giữa": correctness ngang với HYBRID baseline (34.0% vs 33.4%), nhưng retrieval latency **cao hơn gấp 2.4 lần** (1.68s vs 0.71s) dù quét **ít hơn 44% dữ liệu**. Nghịch lý này đến từ CSEP multi-hop overhead.

### Phân tích Latency Component (per-query)

| Component | HYBRID | T-RAG τ=0.15 | T-RAG τ=0.30 |
|-----------|--------|-------------|-------------|
| Router + Encode | 0ms (không có) | ~10ms (nhưng encode **2 lần**) | ~10ms |
| Hop 1 Retrieval | 0.71s (9 tables) | ~0.80s (5.8 tables) | ~0.50s (3.6 tables) |
| Entity Extraction | 0ms | ~0.04s (batched) | ~0.04s |
| Hop 2 Retrieval | 0ms | ~0.80s (5.8 tables lần 2) | ~0.50s |
| **Retrieval tổng** | **0.71s** | **1.68s** | **1.06s** |

> [!IMPORTANT]
> **Phát hiện quan trọng**: Hop 2 gần như tốn gấp đôi Hop 1, nhưng chỉ thêm ~2-5% docs mới (do dedup). Đây là bottleneck lớn nhất.

### Phân tích Quality theo Question Type

| Question Type (count) | HYBRID | T-RAG γ=1 tốt nhất | Ai thắng? |
|----------------------|--------|-------------------|-----------|
| basic (175) | 39.4% | **41.7%** | T-RAG ✅ |
| semantic (125) | 14.4% | **17.6%** | T-RAG ✅ |
| project_related (40) | **20.0%** | 22.5% | ~Ngang |
| conflicting_info (20) | 45.0% | **60.0%** | T-RAG ✅ |
| intra_doc_reasoning (40) | **32.5%** | 30.0% | HYBRID ✅ |
| completeness (20) | 10.0% | **15.0%** | T-RAG ✅ |
| constrained (30) | **33.3%** | 26.7% | HYBRID ✅ |

T-RAG đã **thắng trên 4/7 loại câu hỏi**, đặc biệt mạnh ở semantic và conflicting_info. Nhưng thua ở constrained và intra_document_reasoning — hai loại cần context **nhiều và đầy đủ hơn**.

---

## Phân tích Fairness

### ❌ Không làm (unfair với baseline)

| Thay đổi | Lý do unfair |
|----------|-------------|
| Batch encode tất cả 500 queries cùng lúc | Baseline cũng encode per-query. Trong production, query đến từng cái một. Đây là benchmark trick, không phải cải tiến kiến trúc |
| Upgrade Reranker model | Baselines cũng dùng cùng Reranker. Đổi model = đổi shared component, phải chạy lại toàn bộ baselines |
| Parallel shard search | Baseline cũng sequential loop. Nếu chỉ parallelize T-RAG mà không parallelize baseline → unfair implementation advantage |

### ✅ Sẽ làm (fair — cải tiến kiến trúc thuần túy)

| Thay đổi | Lý do fair |
|----------|-----------|
| Fix double-encode bug | Baseline không bị bug này. Đây là bug fix, không phải trick |
| Smart CSEP (skip Hop 2 khi không cần) | Cải tiến thuật toán routing riêng của T-RAG |
| Tune γ, Dense/Sparse weights | Hyperparameter riêng của T-RAG |
| Adaptive Tau | Innovation cốt lõi mới cho routing |
| Tăng TOP_K_FINAL | Config riêng cho T-RAG v2 (mỗi system có config optimal riêng) |

---

## Thiết kế T-RAG v2

### Chiến lược: **Tập trung Performance (Correctness), giữ Latency tốt hơn HYBRID**

**Mục tiêu dự kiến:**
- Correctness: **35-37%** (so với 34.0% T-RAG v1, 33.4% HYBRID)
- Latency: **0.90-1.05s** (so với 1.07s T-RAG v1 balanced, 1.23s HYBRID)
- Search Space: **~1.5-2.5M** (so với 4.2M HYBRID)

---

### Thay đổi 1: Fix Double-Encode Bug

> [!NOTE]
> Impact dự kiến: **Latency -3~5%**, Quality: không đổi

**Hiện tại** ([csep_retriever.py#L211-L218](file:///network-volume/RAG-/T-RAG_Project/src/retrieval/csep_retriever.py#L211-L218)):
```python
def process_query_hop1(idx, query):
    docs = self.retriever.retrieve(query, ...)  # encode lần 1 bên trong
    emb = self.retriever.router.encoder.encode([query], ...)  # encode lần 2!
    probs = self.retriever.router.clf.predict_proba(emb)[0]
```

**Sửa**: Refactor `retrieve()` để trả thêm `source_probs` và `embedding` đã tính, tránh encode lại:
```python
def retrieve(self, query, top_k=5):
    emb = self.router.encoder.encode([query], normalize_embeddings=True)
    probs = self.router.clf.predict_proba(emb)[0]
    source_probs = {...}
    # ... retrieval logic ...
    return docs, source_probs, emb  # trả thêm 2 giá trị
```

---

### Thay đổi 2: Smart CSEP — Conditional Hop 2

> [!IMPORTANT]
> Impact dự kiến: **Latency -30~40%**, Quality: **-0.5~0%** (gần như không đổi)

**Phát hiện**: Hop 2 tốn ~bằng Hop 1 nhưng chỉ thêm rất ít docs mới (do dedup). Thay vì chạy Hop 2 cho **tất cả** queries:

**Chiến lược Hop 2 có điều kiện:**
1. Sau Hop 1, kiểm tra **chất lượng context** thu được
2. Chỉ chạy Hop 2 khi **Top-1 document có vector distance quá cao** (context kém) HOẶC khi entity extraction tìm được **cross-source entity thực sự** (ví dụ: ticket ID từ source khác)
3. Skip Hop 2 cho ~70-80% queries đơn giản (single-source, context đã tốt)

```python
def _should_run_hop2(self, query_idx, hop1_docs, entities, source_probs):
    # Skip nếu entity extraction trả về NONE
    if entities == "NONE" or len(entities.strip()) < 3:
        return False
    
    # Skip nếu chỉ có 1 active shard (single-source query)
    active_count = sum(1 for p in source_probs.values() if p >= self.tau)
    if active_count < 2:
        return False
    
    # Skip nếu Hop 1 đã có context tốt (top-1 distance thấp)
    if hop1_docs and hop1_docs[0].get("vector_distance", 1.0) < 0.35:
        return False
    
    return True
```

**Tại sao fair**: Đây là cải tiến thuật toán routing nội bộ. Baseline không có CSEP nên không bị ảnh hưởng.

---

### Thay đổi 3: Adaptive Tau (Entropy-based Dynamic Threshold)

> [!NOTE]
> Impact dự kiến: **Latency -5~10%**, Quality: **+0.5~1%**

**Ý tưởng**: Thay vì dùng τ cố định cho mọi query, điều chỉnh dựa trên **entropy** của probability distribution từ Router:

- **Entropy thấp** (Router tự tin, ví dụ 1 source chiếm 80%): tăng τ → quét ít shard → nhanh hơn
- **Entropy cao** (Router không chắc, ví dụ 4-5 sources đều ~20%): giảm τ → quét nhiều shard → coverage tốt hơn

```python
import numpy as np

def adaptive_tau(self, source_probs, tau_base=0.15, alpha=0.08):
    probs = np.array(list(source_probs.values()))
    probs = probs / probs.sum()  # normalize
    
    # Shannon entropy
    H = -np.sum(probs * np.log(probs + 1e-10))
    H_max = np.log(len(probs))  # log(9) ≈ 2.197
    
    # Confidence: 0 (max uncertain) -> 1 (max confident)
    confidence = 1.0 - (H / H_max)
    
    # Khi confident cao → tau tăng (quét ít hơn)
    # Khi confident thấp → tau giảm (quét nhiều hơn)
    tau_eff = tau_base + alpha * (confidence - 0.5)
    
    return max(0.05, min(0.40, tau_eff))  # clamp [0.05, 0.40]
```

**Với τ_base=0.15, α=0.08:**
- Query rất focused (confidence~0.9): τ_eff ≈ 0.18 → ít shard, nhanh
- Query ambiguous (confidence~0.3): τ_eff ≈ 0.13 → nhiều shard, coverage tốt

**Tại sao fair**: Đây là innovation cốt lõi của T-RAG — router thông minh hơn. Baseline không có router.

---

### Thay đổi 4: Tune Hyperparameters

> [!NOTE]
> Impact dự kiến: Quality **+1~3%**, Latency: không đổi

#### 4a. Gamma (γ) = 0.5

Hiện tại benchmark chạy γ=0.0 (không có source bias) và γ=1.0 (linear). Cả hai đều có trade-off:
- γ=0.0: Mọi source bình đẳng → document từ source có P=0.16 (gần tau) được xếp ngang source có P=0.80
- γ=1.0: Source bias tuyến tính → đè mạnh docs từ source phụ
- **γ=0.5: Square-root weighting** → nhẹ nhàng ưu tiên source chính mà không loại bỏ hoàn toàn source phụ

```
P(source)=0.80 → weight = 0.80^0.5 = 0.894  (giữ 89%)
P(source)=0.40 → weight = 0.40^0.5 = 0.632  (giữ 63%)
P(source)=0.16 → weight = 0.16^0.5 = 0.400  (giữ 40%)
```

#### 4b. Dense/Sparse = 0.5/0.5

Hiện tại T-RAG dùng Dense=0.3, Sparse=0.7 (thiên BM25). Nhưng:
- HYBRID baseline dùng 0.5/0.5 đạt 33.4%
- Enterprise queries phức tạp cần semantic search (Dense) nhiều hơn BM25
- Cân bằng 0.5/0.5 cho phép cả hai tín hiệu đóng góp đều

#### 4c. TOP_K_FINAL = 7

- Hiện tại: 5 docs × ~200 tokens = ~1,000 tokens context
- Đề xuất: 7 docs × ~200 tokens = ~1,400 tokens context
- Context window Qwen-14B: 8,192 tokens → vẫn dùng chưa tới **17%** capacity
- VRAM: **Không ảnh hưởng** — `max_model_len=8192` đã cố định, chỉ prompt dài hơn trong cùng window
- Latency: +~0.02-0.03s/query (thêm ~400 tokens prefill, negligible)

> [!WARNING]
> **Về concern VRAM**: Tăng TOP_K_FINAL **không** tăng VRAM. VRAM cho KV cache được allocate dựa trên `max_model_len` (8192) và `gpu_memory_utilization` (0.8), không phải prompt length thực tế. Prompt 1,400 tokens vs 1,000 tokens sử dụng cùng KV cache pool.

---

## Proposed Changes — Cấu trúc T-RAG v2

### File Structure

```
src/trag_v2/
├── __init__.py
├── retriever_v2.py        # [NEW] Retriever với fix double-encode + adaptive tau + tuned weights
├── csep_retriever_v2.py   # [NEW] CSEP với Smart Hop 2 conditional
└── run_benchmark_v2.py    # [NEW] Benchmark runner với v2 config
```

Các module **không thay đổi** (reuse từ v1):
- `src/reranker/reranker.py` — cùng Reranker model (fair)
- `src/generation/generator.py` — cùng LLM, chỉ đổi TOP_K_FINAL
- `src/models/router_inference.py` — cùng PSR Router model

---

### Component: Retriever v2

#### [NEW] [retriever_v2.py](file:///network-volume/RAG-/T-RAG_Project/src/trag_v2/retriever_v2.py)

Dựa trên [retriever.py](file:///network-volume/RAG-/T-RAG_Project/src/retrieval/retriever.py) với các thay đổi:

1. **`retrieve()` trả thêm `source_probs` và `emb`** — xóa bỏ double-encode
2. **Adaptive Tau** — `tau` không còn là constant, mà được tính per-query từ entropy
3. **Fusion weights**: Dense=0.5, Sparse=0.5 (hardcoded trong v2, không đọc từ env)
4. **Gamma = 0.5** (hardcoded)

---

#### [NEW] [csep_retriever_v2.py](file:///network-volume/RAG-/T-RAG_Project/src/trag_v2/csep_retriever_v2.py)

Dựa trên [csep_retriever.py](file:///network-volume/RAG-/T-RAG_Project/src/retrieval/csep_retriever.py) với các thay đổi:

1. **Dùng kết quả `source_probs` và `emb` từ `retrieve()`** — không encode lại
2. **Smart Hop 2 conditional** — chỉ chạy Hop 2 khi thỏa 3 điều kiện:
   - Entity extraction trả về entity thực (không phải NONE)
   - ≥ 2 active shards (query thực sự đa nguồn)
   - Top-1 vector distance > threshold (context chưa đủ tốt)
3. **Dự kiến**: ~70-80% queries skip Hop 2 → latency giảm mạnh

---

#### [NEW] [run_benchmark_v2.py](file:///network-volume/RAG-/T-RAG_Project/src/trag_v2/run_benchmark_v2.py)

Benchmark runner với config v2 hardcoded:

```python
# T-RAG v2 Config (hardcoded, không đọc env)
TAU_BASE = 0.15
TAU_ALPHA = 0.08
GAMMA = 0.5
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 7
ENABLE_CSEP = True
SMART_HOP2 = True  # Conditional Hop 2
```

---

## Tóm tắt Impact dự kiến

| Metric | T-RAG v1 (best) | HYBRID | T-RAG v2 (dự kiến) |
|--------|-----------------|--------|-------------------|
| Correctness | 34.0% | 33.4% | **35-37%** |
| Completeness | 43.5% | 44.1% | **44-46%** |
| Refused | 19.2% | 18.0% | **17-19%** |
| Latency | 1.07s (τ=0.15) | 1.23s | **0.90-1.05s** |
| Search Space | 2,354,016 | 4,213,106 | **~2.0-2.5M** |

### Nguồn gốc cải thiện:
- **Correctness +1~3%**: TOP_K_FINAL 7 (thêm context) + balanced Dense/Sparse (retrieval tốt hơn) + soft gamma (ranking tốt hơn)
- **Latency -10~15%**: Smart Hop 2 conditional (skip ~70% Hop 2) + fix double-encode
- **Refused -1~2%**: Thêm 2 docs context giúp LLM tự tin hơn, bớt từ chối

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Bạn muốn focus chính vào **Performance (Correctness)** hay **Latency**? Thiết kế hiện tại cân bằng cả hai, nhưng nếu muốn aggressive hơn về một phía, tôi có thể điều chỉnh (ví dụ: skip CSEP hoàn toàn để latency ~0.70s, hoặc giảm tau xuống 0.10 để quality cao hơn nhưng latency ~1.2s).

> [!NOTE]
> **Q2**: TOP_K_FINAL = 7 — bạn OK với việc T-RAG v2 dùng 7 docs trong khi baselines dùng 5? Mỗi system có config optimal riêng là fair, nhưng nếu bạn muốn strict so sánh thì giữ 5.

---

## Verification Plan

### Automated Tests
```bash
# Chạy T-RAG v2 benchmark
python src/trag_v2/run_benchmark_v2.py --questions questions.jsonl --output results_v4/trag_v2.jsonl

# Chạy evaluation
python src/evaluation/metrics_based_eval.py \
  --questions questions.jsonl \
  --answers results_v4/trag_v2.jsonl \
  --output results_v4/eval_trag_v2.json

# So sánh kết quả
python src/evaluation/generate_report.py results_v4
```

### Manual Verification
- So sánh bảng kết quả T-RAG v2 vs v1 vs HYBRID
- Kiểm tra per-question-type breakdown
- Xác nhận latency và search space
- Update slide nếu kết quả tốt hơn
