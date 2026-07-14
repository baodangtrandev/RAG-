# Code Review: T-RAG Pipeline - Trước khi Implementation

## Đánh giá tổng thể

Codebase hiện tại có **nền tảng tốt** (PSR, DB sharding, SW-RRF đều hợp lý), nhưng có **một số vấn đề kỹ thuật nghiêm trọng** nếu không sửa trước thì phần implementation mới sẽ không hoạt động đúng. Báo cáo này chia theo mức độ ưu tiên.

---

## 🔴 Vấn đề NGHIÊM TRỌNG (Phải sửa trước khi code thêm)

### 1. `retriever.py` — Dùng field `text` nhưng schema LanceDB lưu field `content`

**Vấn đề:** Trong [`ingest.py`](file:///network-volume/RAG-/scripts/ingest.py) (dòng 101), dữ liệu được nạp vào LanceDB với key `"content"`, đúng theo [`lance_schema.py`](file:///network-volume/RAG-/src/ingestion/lance_schema.py) (`content: str`). Tuy nhiên, trong [`retriever.py`](file:///network-volume/RAG-/src/retrieval/retriever.py) (dòng 90), code lại lấy:
```python
"text": doc.get("text", doc.get("content", "")),
```
Đây là lớp bọc fallback. Thực ra field `text` không bao giờ tồn tại, code chỉ đang lúc nào cũng dùng fallback sang `content`. Không nguy hiểm vào lúc này, nhưng key chuẩn ra phía ngoài là `"text"` trong khi DB lưu `"content"` - sẽ gây nhầm lẫn khi các module mới (Reranker, Generator, CSEP) cần đọc field này. **Cần chuẩn hóa.**

**Đề xuất:** Sửa dòng 90 trong `retriever.py` thành:
```python
"content": doc.get("content", ""),
```
Và cập nhật `__main__` block để dùng `doc['content']` thay cho `doc['text']`.

---

### 2. `router_inference.py` — Lỗi biến bị mất sau khi update

**Vấn đề:** Sau khi em cập nhật `__init__`, có một bug nhỏ. Dòng 20:
```python
classes_path = os.path.join(model_dir, "psr_classes.json")
```
Ở đây `model_dir` vẫn lấy từ biến argument cũ (chưa được reassign). Sau khi sửa, `self.model_dir` đã đúng nhưng `model_dir` local variable thì vẫn ổn vì được assign ngay ở dòng trên. Tuy nhiên, **cần kiểm tra lại** luồng assign `self.model_dir` vs `model_dir` local để đảm bảo `os.path.join(self.model_dir, ...)` nhất quán ở tất cả các nơi.

**Đề xuất:** Dùng `self.model_dir` thay cho `model_dir` trong toàn bộ `__init__` để tránh nhầm lẫn.

---

### 3. `.env` — Vẫn còn thông tin proxy cũ (Risk bảo mật)

**Vấn đề:** File [`.env`](file:///network-volume/RAG-/.env) hiện vẫn còn:
```
LLM_API_BASE_URL="https://proxy-metrolist.nguyendinhbang53az.workers.dev"
LLM_API_KEY="9HdE8Mgvn-Kfd5a6vun4dKyuboA7Hcupr5gTE4li"
```
Anh/chị đã quyết định dùng **Local LLM**, không cần proxy nữa. Giữ API key không dùng trong file `.env` là một **rủi ro bảo mật** (nếu file này vô tình bị commit lên git). Hơn nữa file `.env.example` cũng đang outdated hoàn toàn.

**Đề xuất:** Xóa 2 dòng proxy/key cũ và đồng bộ lại `.env.example`.

---

### 4. `retriever.py` — Block `__main__` hardcode đường dẫn tuyệt đối

**Vấn đề:** Dòng 112:
```python
db_path = "/network-volume/RAG-/data/lancedb"
```
Hardcode đường dẫn tuyệt đối sẽ **làm hỏng** script khi chạy trên máy khác hoặc khi anh/chị thay đổi `RAG_DB_URI` trong `.env`. Block `__main__` phải đọc từ `.env`.

**Đề xuất:** Sửa thành:
```python
from dotenv import load_dotenv
load_dotenv()
db_path = os.environ.get("RAG_DB_URI", "data/lancedb")
retriever = EnterpriseRetriever()  # Tự load từ env, không cần truyền arg
```

---

## 🟡 Vấn đề THIẾT KẾ (Cần thống nhất trước khi code module mới)

### 5. CSEP trong Implementation Plan — Thiếu rõ ràng về `top_k` cho từng Hop

**Vấn đề:** Plan nói CSEP thực hiện "Hop 1 → Entity Extraction → Hop 2" nhưng chưa định nghĩa:
- Hop 1 lấy bao nhiêu `anchor_docs`?
- Sau khi gộp kết quả 2 hop, cross-encoder rerank trên bao nhiêu docs?

Nếu không thiết kế trước, các module CSEP, Reranker và Generator sẽ có `top_k` không thống nhất.

**Đề xuất:** Bổ sung 2 biến vào `.env`:
```
RAG_TOP_K_RETRIEVE="20"    # Số docs lấy ra trước Reranker
RAG_TOP_K_FINAL="5"        # Số docs cuối cùng đưa vào LLM (sau Reranker)
```

---

### 6. CSEP — Entity Extraction gọi LLM trong Retrieval Stage là nguy hiểm cho performance

**Vấn đề:** Nếu CSEP kích hoạt cho **toàn bộ** câu hỏi và mỗi lần Hop 1 đều cần gọi LLM để extract entity, thì với 500 câu hỏi trong benchmark, ta sẽ gọi LLM **500 lần** tuần tự trong vòng lặp Retrieval — điều này **phá vỡ hoàn toàn** chiến lược Stage-based Batching đã đề ra. Throughput sẽ rất tệ.

**Đề xuất (quan trọng):** Tách CSEP Entity Extraction thành một **Batch Stage riêng**:
1. Chạy Retrieval Hop 1 cho **tất cả 500 queries** → thu được `500 × anchor_docs`.
2. Chạy **một lần** LLM batch với 500 prompts entity extraction.
3. Chạy Retrieval Hop 2 cho tất cả 500 queries với entity đã trích xuất.

Thiết kế này mới đúng với Stage-based Batching và tận dụng được H100.

---

### 7. Plan hiện tại — Thiếu module Reranker riêng cho Cross-Encoder

**Vấn đề:** Plan Giai đoạn 1 đề cập "hàm `rerank_batch`" nhưng không nói rõ: Cross-Encoder model này sẽ được nạp lên GPU riêng, hay dùng chung với vLLM? Nếu vLLM chiếm 80% VRAM (~32GB), còn lại ~8GB cho Cross-Encoder (có thể nặng đến 500MB-1GB). Cần xác nhận Cross-Encoder chạy **sau** khi vLLM CSEP batch entity-extraction xong để không bị OOM.

**Đề xuất:** Thêm biến `RERANKER_MODEL` vào `.env` (ví dụ: `cross-encoder/ms-marco-MiniLM-L-6-v2`). Thiết kế thứ tự nạp model: Cross-Encoder nạp trước, vLLM nạp sau (để dễ kiểm soát VRAM).

---

### 8. `data_analysis_report.md` — Tài liệu phân tích dữ liệu nhưng không được dùng làm input

**Vấn đề:** Có file `data_analysis_report.md` ở thư mục gốc nhưng không file nào import hay reference nó. Chỉ là tài liệu đọc tham khảo, không ảnh hưởng đến pipeline.

---

## 🟢 Những điểm tốt, GIỮ NGUYÊN

1. **Physical Sharding (LanceDB):** Thiết kế tách 9 bảng độc lập theo source type là cực kỳ đúng đắn. Đã có 9 bảng `.lance` trong `data/lancedb/` → **Sẵn sàng để dùng**.
2. **PSR Model:** Đã train, `psr_router.joblib` + `psr_classes.json` có sẵn. Hit Rate 94.9% tại `tau=0.15` là rất tốt → **Không cần touch**.
3. **SW-RRF Logic:** Công thức toán học trong `retriever.py` (dòng 76-84) chính xác theo Proposal → **Giữ nguyên**.
4. **FP16 Optimization:** Cả `ingest.py`, `train_psr.py`, `evaluate_psr.py` đều đã dùng `torch.float16` → **Nhất quán, tốt**.
5. **Makefile + `.flake8` + `pyproject.toml`:** Hạ tầng CI/CD cơ bản đã sẵn sàng.

---

## Tóm tắt: Checklist cần làm TRƯỚC khi code module mới

| # | Vấn đề | Độ ưu tiên | File |
|---|--------|-----------|------|
| 1 | Chuẩn hóa `content` vs `text` field | 🔴 Cao | `retriever.py` |
| 2 | Nhất quán `self.model_dir` trong `router_inference.py` | 🔴 Cao | `router_inference.py` |
| 3 | Xóa thông tin proxy/key cũ, cập nhật `.env.example` | 🔴 Cao | `.env`, `.env.example` |
| 4 | Sửa hardcode path trong `__main__` của `retriever.py` | 🔴 Cao | `retriever.py` |
| 5 | Thêm `RAG_TOP_K_RETRIEVE`, `RAG_TOP_K_FINAL`, `RERANKER_MODEL` vào `.env` | 🟡 Trung bình | `.env` |
| 6 | Thiết kế lại CSEP thành **Batch Stage** (không gọi LLM tuần tự) | 🟡 Trung bình | Implementation Plan |
| 7 | Cập nhật `.env.example` để đồng bộ | 🟡 Trung bình | `.env.example` |
