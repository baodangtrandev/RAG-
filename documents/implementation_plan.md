# Kế hoạch triển khai T-RAG Pipeline (Phiên bản Tối ưu H100 Offline Batching)

Bản kế hoạch này mô tả kiến trúc và các bước triển khai hệ thống T-RAG, được thiết kế đặc biệt để **tối đa hóa hiệu năng (Throughput)** trên **GPU H100 (40GB - 80GB VRAM)**, phục vụ mục tiêu chạy tập dữ liệu **EnterpriseRAG-Bench**.

## User Review Required

> [!IMPORTANT]
> Dựa trên góp ý cực kỳ chính xác của bạn về vLLM Offline Batching, tôi đã thiết kế lại luồng thực thi (Execution Flow) từ dạng tuần tự (Sequential) sang dạng **Theo giai đoạn (Stage-based Batching)**. Mời bạn xem xét!

---

## 1. Phân tích: Vì sao bạn/người bạn đó nói đúng 100%?

Người khuyên bạn câu đó là một kỹ sư hệ thống rất am hiểu! 
1. **Overhead của HTTP API:** Việc chạy server và gọi qua HTTP (dù là `localhost`) yêu cầu JSON Serialization, Deserialization, HTTP header parsing, và quan trọng nhất là bị giới hạn bởi connection pool.
2. **Quyền năng của Offline Batching:** `vLLM` sinh ra là để tối ưu hóa Memory Bound qua cơ chế **PagedAttention**. Khi dùng class `LLM(model="...")` và truyền vào list 500 prompts cùng lúc (`llm.generate(prompts)`), engine sẽ tự động xếp lịch (schedule), nhét đầy các token vào các block VRAM trống của H100, tạo ra Batch Size khổng lồ (vài trăm câu hỏi cùng lúc). Tốc độ có thể **nhanh gấp 10 - 50 lần** so với việc for-loop từng câu qua API.

## 2. Kiến trúc Data Flow Mới (Tối đa hóa H100)

Để xài được Offline Batching, chúng ta **KHÔNG THỂ** làm theo cách truyền thống (for-loop từng câu: bóc tách $\rightarrow$ tìm kiếm $\rightarrow$ rerank $\rightarrow$ trả lời). Vì như thế batch_size ở LLM luôn = 1.
Chúng ta sẽ thiết kế Pipeline xử lý **toàn bộ 500 câu hỏi cùng lúc** qua từng trạm (Stage):

### Stage 1: Batch Query Parsing (LLM)
- Đưa cùng lúc 500 câu hỏi trong `questions.jsonl` vào vLLM thông qua `llm.generate(prompts)`.
- vLLM trả về 500 kết quả JSON (chứa `source_type`, keyword...).

### Stage 2: Batch Hybrid Retrieval (LanceDB)
- Dùng vòng lặp (hoặc Async) truy vấn 500 queries vừa bóc tách vào **LanceDB** (áp dụng luôn Metadata Filtering + Hybrid BM25/Vector).
- Kết quả trả về danh sách 500 x Top 50 documents.

### Stage 3: Batch Temporal Reranking (Cross-Encoder)
- Gom 500 x 50 = 25.000 cặp (Query, Document) ném vào model `BGE-Reranker-v2-m3` với `batch_size=256` (GPU H100 dư sức tính 25k phép tính này trong vài giây).
- Áp dụng công thức phạt thời gian **Time Decay** ($e^{-\lambda \Delta t}$) cho toàn bộ ma trận điểm.
- Cắt lại Top 10 documents cho mỗi câu.

### Stage 4: Batch Answer Generation (LLM)
- Ghép 500 câu hỏi gốc với Top 10 documents tương ứng tạo thành 500 prompts khổng lồ.
- Đưa vào `llm.generate(prompts)` lần 2. H100 sẽ xả toàn bộ sức mạnh để sinh ra 500 câu trả lời cùng lúc.

---

## 3. Lựa chọn Tech Stack (Chốt)

1. **Storage (Ingestion):** Chỉ sử dụng duy nhất **LanceDB** (Hỗ trợ Native Vector Search + BM25 Tantivy + Hybrid RRF + Metadata Filtering).
2. **LLM Engine:** `vllm.LLM` chạy in-memory (Offline Batching mode).
3. **Reranker Engine:** `sentence-transformers` (Chạy batch processing).
4. **Data:** File `all_documents.zip` đã có sẵn, chúng ta sẽ viết script giải nén và nạp thẳng vào LanceDB.

## 4. Kế hoạch Code (Các files)

1. `ingest.py`: Đọc file `all_documents.zip`, tạo embedding, và nạp vào LanceDB.
2. `vllm_engine.py`: Wrapper chứa object `LLM` và các hàm helper gọi `generate()`.
3. `trag_pipeline.py`: Chứa logic của 4 Stage trên. Xử lý input là list, output là list.
4. `run_benchmark.py`: Script khởi chạy toàn bộ luồng, ghi log throughput, và xuất ra `answers.jsonl` chuẩn hóa.

Mọi thứ đã rất logic và sẵn sàng cho H100! Đợi bạn xác nhận để chúng ta bắt đầu code `ingest.py` tải dữ liệu vào DB!
