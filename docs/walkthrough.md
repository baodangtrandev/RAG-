# Walkthrough: T-RAG Pipeline Implementation

## Tóm tắt

Toàn bộ T-RAG pipeline đã được hoàn thiện theo đúng **Final Implementation Plan** đã được duyệt. Tất cả **28/28 unit tests đều PASS**.

---

## Những gì đã được thực hiện

### Phần A: Pre-Flight Fixes (4 lỗi đã sửa)

| Fix | File | Thay đổi |
|-----|------|----------|
| 1 | [`router_inference.py`](file:///network-volume/RAG-/src/models/router_inference.py) | `model_dir` → `self.model_dir` trong `__init__` (tránh NullPointerError khi argument là `None`) |
| 2 | [`retriever.py`](file:///network-volume/RAG-/src/retrieval/retriever.py) | Key `"text"` → `"content"` nhất quán với LanceDB schema |
| 3 | [`retriever.py`](file:///network-volume/RAG-/src/retrieval/retriever.py) | Bỏ hardcode `/network-volume/RAG-/...` trong `__main__`, đọc từ `.env` |
| 4 | [`.env`](file:///network-volume/RAG-/.env) + [`.env.example`](file:///network-volume/RAG-/.env.example) | Xóa API key proxy cũ; thêm `RAG_TOP_K_RETRIEVE`, `RAG_TOP_K_FINAL`, `RERANKER_MODEL`, `RERANKER_THRESHOLD`, `VLLM_GPU_MEMORY_UTILIZATION` |

---

### Phần B: Các module mới được tạo

#### Giai đoạn 1: [`src/reranker/reranker.py`](file:///network-volume/RAG-/src/reranker/reranker.py)
- Class `CrossEncoderReranker`
- Batch inference GPU: tạo flat list tất cả `(query, doc)` pairs → một lần `model.predict()` → reshape lại
- Lọc docs theo `RERANKER_THRESHOLD`, sort theo score giảm dần
- Flag `is_unanswerable=True` khi không còn doc nào
- **Logging:** INPUT count, inference time + throughput, OUTPUT count

#### Giai đoạn 2: [`src/generation/generator.py`](file:///network-volume/RAG-/src/generation/generator.py)
- Class `VLLMGenerator` với vLLM Offline Batching
- Phân loại answerable/unanswerable *trước* khi gọi LLM — queries unanswerable không tốn token
- Một lần `llm.generate(N prompts)` cho tất cả answerable queries
- **Logging:** số prompts, inference time, throughput (tokens/s)

#### Giai đoạn 3: [`src/retrieval/csep_retriever.py`](file:///network-volume/RAG-/src/retrieval/csep_retriever.py)
- Class `CSEPRetriever` với 4 sub-stages theo đúng thiết kế Batch Stage:
  - **Sub-A:** Hop 1 retrieval cho tất cả N queries
  - **Sub-B:** Batch entity extraction — **một lần gọi LLM với N prompts**
  - **Sub-C:** Hop 2 retrieval với augmented queries
  - **Sub-D:** Merge + dedup by `doc_id` + re-sort
- Toggle `ENABLE_CSEP_FOR_ALL` từ `.env`
- **Logging:** mỗi sub-stage log input size + elapsed time

#### Giai đoạn 4: [`src/run_benchmark.py`](file:///network-volume/RAG-/src/run_benchmark.py)
- CLI với `typer`: tất cả hyperparameters có thể override qua CLI args
- Hỗ trợ cả `.parquet` và `.jsonl` input
- Flag `--limit` để test nhanh
- Output: `answers.jsonl` theo format chuẩn EnterpriseRAG-Bench
- Summary cuối: throughput, tỷ lệ unanswerable, thời gian từng stage

---

## Test Results

```
28 passed in 10.23s
├── tests/test_csep.py          8/8   ✅
├── tests/test_generator.py    10/10  ✅
├── tests/test_reranker.py      5/5   ✅
└── tests/test_markdown_chunker 5/5   ✅
```

---

## Cách chạy Benchmark

```bash
# Activate env
conda activate trag

# Chạy với config mặc định (Llama 3.1 8B, CSEP=True)
python src/run_benchmark.py

# So sánh với Qwen 2.5 14B
python src/run_benchmark.py --model Qwen/Qwen2.5-14B-Instruct --output answers_qwen.jsonl

# Test nhanh với 10 câu hỏi đầu
python src/run_benchmark.py --limit 10

# Tắt CSEP để benchmark baseline
python src/run_benchmark.py --no-csep --output answers_no_csep.jsonl

# Benchmark với tau khác
python src/run_benchmark.py --tau 0.2 --top-k-final 3
```

---

## Logging Standard

Tất cả module đều in log theo format:
```
2026-07-14 11:20:00 | INFO | src.reranker.reranker | [Reranker] INPUT: 500 queries | 10000 pairs (query, doc) | threshold=0.0
2026-07-14 11:20:02 | INFO | src.reranker.reranker | [Reranker] ⚡ Inference done: 10000 pairs | 2.10s | 4761 pairs/s
2026-07-14 11:20:02 | INFO | src.reranker.reranker | [Reranker] OUTPUT: 0/500 unanswerable | total_dropped=0 docs
```

Khi hệ thống có lỗi, log sẽ giúp xác định ngay vấn đề ở Stage nào.
