# Kế hoạch triển khai T-RAG Pipeline (Benchmark & Academic Setup)

Bản kế hoạch này mô tả kiến trúc và các bước triển khai hệ thống T-RAG, được thiết kế đặc biệt cho mục tiêu **chạy benchmark (EnterpriseRAG-Bench)** và **viết paper**, chạy trên **1 node GPU H100 (40GB - 80GB VRAM)**.

Vì mục tiêu là nghiên cứu và công bố khoa học (không phải production microservices), kiến trúc sẽ ưu tiên:
- **Dễ tái tạo (Reproducibility):** Hạn chế tối đa các dependencies phức tạp (như Docker services riêng rẽ).
- **Tốc độ (Throughput):** Tối ưu vRAM và batching để chạy benchmark nhanh.
- **Minh bạch (Transparency):** Code native Python để dễ dàng đo lường độ trễ (latency), ablation studies, và log lại các bước trung gian phục vụ viết báo.

## User Review Required

> [!IMPORTANT]
> Dưới đây là các quyết định (Tech Stack Decisions) được lựa chọn dựa trên ngữ cảnh Benchmark & H100 của bạn. Xin hãy review và phê duyệt để tôi bắt đầu code.

## Tech Stack Decisions (Khuyến nghị cho Benchmark/Paper)

**1. Database Stack (Local & Lightweight)**
- **Relational DB & Sparse Search:** Sử dụng **SQLite** kết hợp module **FTS5** (Full-Text Search).
  - *Lý do:* Không cần cài đặt server (như Postgres/Elasticsearch), chỉ là một file local, cực kỳ dễ share code để người khác tái tạo kết quả paper. FTS5 hỗ trợ thuật toán BM25 trực tiếp trong SQLite, giải quyết luôn cả bài toán Metadata Filtering và Sparse Search trong cùng một câu query SQL.
- **Vector DB:** Sử dụng **LanceDB** hoặc **Qdrant (Local/In-memory mode)**.
  - *Lý do:* Chạy trực tiếp trong memory/disk bằng Python mà không cần spin up Docker container. Xử lý 500k vectors rất mượt.

**2. LLM Stack (Tối ưu cho H100 40-80GB)**
Với H100, tài nguyên tính toán rất mạnh nhưng VRAM (nếu là 40GB) cần được quy hoạch kỹ:
- **Engine Suy luận:** Chắc chắn phải dùng **vLLM** (hoặc TensorRT-LLM) để đạt throughput tối đa (cần thiết khi chạy benchmark với hàng nghìn câu hỏi).
- **Mô hình (Models):**
  - *Parser & Generator:* Chỉ nên dùng 1 mô hình duy nhất (ví dụ: `Meta-Llama-3-8B-Instruct` hoặc `Qwen2.5-7B-Instruct`) cho cả việc parse câu hỏi và sinh câu trả lời để tiết kiệm VRAM (chiếm khoảng 15-18GB ở bf16).
  - *Reranker:* Dùng thư viện `sentence-transformers` hoặc `vLLM` cho `BGE-Reranker-v2-m3` (chiếm khoảng 4-6GB).
  - *Embedder (nếu cần tự nhúng):* `text-embedding-3-large` (như bài báo) hoặc model open-source `BGE-m3`. Nếu bài báo đã cung cấp sẵn file nhúng, ta có thể bỏ qua việc load model này.
  *-> Tổng VRAM dự kiến: ~25GB, chạy rất an toàn và thoải mái trên H100 40GB, cho phép Batch Size lớn.*

**3. Framework**
- **Native Python + asyncio**: KHÔNG dùng LangChain hay LlamaIndex.
  - *Lý do:* Khi viết paper, bạn cần đo chính xác thời gian ở từng module (Ablation study) và ghi log mọi thứ để phân tích lỗi. Việc dùng Native Python giúp bạn kiểm soát hoàn toàn data flow và thuật toán (nhất là cái công thức Decay Function bạn tự định nghĩa).

---

## Proposed Architecture & Changes (T-RAG Core)

Hệ thống sẽ được chia thành 4 phân hệ (Modules) chính, bám sát các fix từ `TRAG-N.md`:

### 1. Phân hệ Ingestion & Lưu trữ (Storage Layer)
- **SQLite Database:** Bảng `documents` chứa `doc_id`, `metadata` (JSON), và `content`. Index FTS5 ảo trên bảng này.
- **Vector Index:** LanceDB/Qdrant map `doc_id` với Vector Embedding.

### 2. Phân hệ Phân tích Truy vấn (Query Parser với Soft-Filtering)
- **Entity Extraction (vLLM):** Prompt LLM trả về JSON chứa `entities`, `timeframe`, `doc_type`.
- **Soft-Filtering & Fallback Logic (SQLite):** 
  - Translate JSON thành SQLite Query với logic LIKE/FTS.
  - **Fallback:** Nếu `COUNT(doc_id) < 50`, tự động thả lỏng các điều kiện lọc trong SQL (bỏ filter thời gian, bỏ filter tác giả) cho đến khi lấy đủ ~500 Candidate IDs.

### 3. Phân hệ Tìm kiếm Kết hợp (Hybrid Retriever)
- Sparse Search: Query bằng FTS5 (BM25) trên SQLite, lấy điểm `bm25_score`.
- Dense Search: Query Qdrant/LanceDB với ID nằm trong danh sách Candidate IDs. Lấy `cosine_score`.
- Merge bằng RRF (Reciprocal Rank Fusion).

### 4. Phân hệ Reranker (Temporal & Diversity Decay)
- **Relevance:** Chạy qua `BGE-Reranker-v2` để lấy điểm gốc.
- **Áp dụng Decay Function:**
  - `Final_Score = Relevance * exp(-λ * Δt) * Diversity_Penalty`
  - Logic tính `Diversity_Penalty` sử dụng thuật toán MMR (Maximal Marginal Relevance) chạy trên CPU (numpy/torch-cpu) vì ma trận nhỏ (top 50).
- Lấy Top 10 đưa vào vLLM để sinh câu trả lời cuối cùng.

---

## Các File Sẽ Được Tạo (Cấu trúc Codebase dự kiến)

Chúng ta sẽ thiết kế codebase dạng script chạy benchmark:

#### `d:\Projects\RAG-\TRAG\database.py`
Xử lý setup SQLite (FTS5) và LanceDB/Qdrant. Gồm hàm load bộ dữ liệu 500k docs của EnterpriseRAG-Bench.

#### `d:\Projects\RAG-\TRAG\llm_engine.py`
Khởi tạo và quản lý `vLLM` instance. Chứa các helper func để generate text và parse JSON.

#### `d:\Projects\RAG-\TRAG\retriever.py`
Thực thi Fallback SQL, Dense Search, Sparse Search và RRF.

#### `d:\Projects\RAG-\TRAG\reranker.py`
Chạy Cross-encoder model và tính công thức Decay/MMR.

#### `d:\Projects\RAG-\TRAG\benchmark_runner.py`
File thực thi chính, load tập câu hỏi (`questions.jsonl`), chạy luồng T-RAG (Parser -> Retrieve -> Rerank -> Generate) cho từng câu, đo đạc thời gian (latency) và ghi kết quả ra file `answers.jsonl` chuẩn format của paper.

## Verification Plan

### Automated Benchmark Test
- Chạy thử `benchmark_runner.py` trên tập nhỏ (10-20 câu hỏi) của EnterpriseRAG-Bench.
- Log chi tiết các chỉ số:
  - Thời gian parse.
  - Số lượng tài liệu lọc được ở bước Fallback.
  - Thời gian reranking.
  - Điểm Rerank trước và sau khi áp dụng Time Decay.

### Manual Verification
- Bạn kiểm tra file output `answers.jsonl` đảm bảo đúng định dạng yêu cầu của EnterpriseRAG-Bench leaderboards.
