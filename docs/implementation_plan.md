# Implementation Plan: T-RAG Pipeline Completion (H100 Optimized)

Mục tiêu: Hoàn thiện các module còn thiếu của kiến trúc T-RAG (Targeted RAG) theo tài liệu `proposal.md` và paper `EnterpriseRAG-Bench`. Đặc biệt, thiết kế luồng xử lý tập trung vào việc tối đa hóa hiệu năng (Throughput/Latency) và độ chính xác (Accuracy) trên máy chủ sử dụng **GPU H100 (40GB VRAM) & 32GB RAM**.

## User Review Required

> [!IMPORTANT]
> **Tối ưu Accuracy/Latency trên H100:** 
> Dựa trên paper EnterpriseRAG-Bench và cấu hình phần cứng của anh/chị, để đạt được cả tốc độ và độ chính xác, chúng ta sẽ áp dụng chiến lược **Stage-based Batching với vLLM** thay vì vòng lặp tuần tự (Sequential). Điều này có nghĩa là thay vì hỏi LLM từng câu, ta sẽ gom hàng trăm câu hỏi đưa vào `vLLM` cùng lúc để tận dụng PagedAttention của GPU.
> **Đề xuất LLM Model:** Với VRAM 40GB, model tối ưu nhất hiện nay để triển khai Local trên `vLLM` là **`Meta-Llama-3.1-8B-Instruct`** hoặc **`Qwen2.5-14B-Instruct`**. Chúng có khả năng xử lý ngữ cảnh dài (Long Context) rất tốt, tốn khoảng 16-20GB VRAM cho trọng số (weights), phần VRAM còn lại (20GB) hoàn toàn đủ để chứa KV Cache cực lớn cho batch size hàng trăm câu hỏi.

---

## Proposed Changes

Quá trình triển khai sẽ được chia thành 4 giai đoạn, mỗi giai đoạn sẽ đi kèm với bài test độc lập.

### Giai đoạn 1: Module Reranker (Dynamic Thresholding)

Module này giúp đánh giá lại mức độ liên quan của tài liệu được trả về từ SW-RRF và query gốc, loại bỏ tài liệu nhiễu.

#### [NEW] [reranker.py](file:///network-volume/RAG-/src/reranker/reranker.py)
*   Sử dụng `CrossEncoder` từ `sentence-transformers` (ví dụ: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
*   Hàm `rerank` tính toán điểm số. Những tài liệu có điểm `< threshold` sẽ bị loại bỏ. Khả năng chạy batch inference trên GPU để xử lý cực nhanh.
*   Nếu không còn tài liệu nào sau khi lọc, trả về cờ "Unanswerable".

#### [NEW] [test_reranker.py](file:///network-volume/RAG-/tests/test_reranker.py)
*   Unit test cho Reranker.

---

### Giai đoạn 2: Thuật toán Cross-Source Entity Propagation (CSEP)

Đây là thuật toán để giải quyết các truy vấn đa nguồn phức tạp. **Theo yêu cầu của anh/chị, CSEP sẽ được kích hoạt cho toàn bộ câu hỏi theo mặc định.**

#### [NEW] [csep_retriever.py](file:///network-volume/RAG-/src/retrieval/csep_retriever.py)
*   **Thêm Header Comment:** Ghi chú rõ ở đầu file về cơ chế hoạt động của CSEP và cách sử dụng biến môi trường.
*   **Logic Kích hoạt (Toggle):** Đọc biến `ENABLE_CSEP_FOR_ALL` từ file `.env`.
    *   Nếu `True` (Mặc định): Mọi câu hỏi đều chạy qua luồng Multi-hop.
    *   Nếu `False`: Chỉ kích hoạt khi Router trả về từ 2 nguồn trở lên có xác suất lớn hơn `tau`.
*   **Luồng hoạt động Multi-hop:**
    1.  **Hop 1:** Tìm kiếm trên shard có xác suất cao nhất.
    2.  Dùng LLM (qua batching) trích xuất Entities.
    3.  **Hop 2:** Nối Entities vào query và tìm kiếm trên các shard còn lại.
    4.  Gộp kết quả của 2 hop.

#### [NEW] [test_csep.py](file:///network-volume/RAG-/tests/test_csep.py)
*   Unit test cho module CSEP.

---

### Giai đoạn 3: Module Generation (LLM Integration)

Module kết nối với LLM để tạo ra câu trả lời cuối cùng dựa trên các tài liệu đã được lọc bởi Reranker.

#### [NEW] [generator.py](file:///network-volume/RAG-/src/generation/generator.py)
*   Xây dựng class `RAGGenerator`.
*   Tích hợp Client gọi LLM (tương thích với Local vLLM server hoặc API proxy đã định nghĩa trong `.env`).
*   Xử lý logic fallback nếu không có documents.

#### [NEW] [test_generator.py](file:///network-volume/RAG-/tests/test_generator.py)
*   Unit test cho Generator (sử dụng mock API).

---

### Giai đoạn 4: Tích hợp Hệ thống (End-to-End Pipeline)

Kết nối tất cả các thành phần lại với nhau để tạo thành một đường ống hoàn chỉnh.

#### [NEW] [main.py](file:///network-volume/RAG-/src/main.py)
*   Xây dựng class `TRAGPipeline` điều phối luồng:
    `Batch Query` $\rightarrow$ `Router` $\rightarrow$ `CSEP (Dựa trên env)` $\rightarrow$ `Cross-Encoder Reranker` $\rightarrow$ `Batch LLM Generation`.
*   Tích hợp CLI bằng `typer`.

## Verification Plan

### Automated Tests
- Chạy toàn bộ test suite bằng lệnh `pytest tests/` để đảm bảo các module độc lập hoạt động đúng.

### Manual Verification
- Test chạy CLI với cờ `ENABLE_CSEP_FOR_ALL=True` và `ENABLE_CSEP_FOR_ALL=False` để đảm bảo luồng routing chuyển hướng đúng mong đợi.
- So sánh kết quả sinh ra với các baseline test case có sẵn.
