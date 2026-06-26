# Hướng dẫn Deploy T-RAG Benchmark lên Server H100

Toàn bộ Pipeline của T-RAG đã được code xong và tối ưu hóa 100% cho GPU H100 với kiến trúc **Stage-based Batching**. Đây là hướng dẫn để bạn mang bộ source này lên cluster chạy.

## 1. Cấu trúc Source Code

Các file code được đặt trong thư mục `d:\Projects\RAG-\TRAG\`:
- `ingest.py`: Đọc 500k file `.txt` từ `all_documents.zip`, tính vector nhúng bằng `BAAI/bge-m3`, và nạp vào LanceDB (có cả Vector Index và Full-Text Search Index Tantivy).
- `vllm_engine.py`: Quản lý VRAM và khởi tạo `vllm.LLM`, bọc các hàm PagedAttention Batching.
- `trag_pipeline.py`: Chứa class `TRAGPipeline` thực thi 4 trạm xử lý:
  - Stage 1: Batch LLM Extract Metadata
  - Stage 2: Batch LanceDB Hybrid Search (Vector + BM25)
  - Stage 3: Batch CrossEncoder Reranking (có áp dụng hàm phạt Time Decay)
  - Stage 4: Batch LLM Answer Generation
- `run_benchmark.py`: Script chính khởi động toàn bộ, load 500 câu hỏi, chạy qua pipeline và ghi kết quả ra `answers.jsonl` cực nhanh.

## 2. Chuẩn bị Môi trường trên H100 (Linux)

Mang toàn bộ thư mục `TRAG`, `EnterpriseRAG-Bench` và file `all_documents.zip` lên server.

Cài đặt các thư viện lõi (chú ý: vLLM yêu cầu Linux/CUDA):
```bash
pip install lancedb tantivy sentence-transformers vllm openai pydantic pyarrow pandas
```

> [!TIP]
> Hãy đảm bảo PyTorch được cài đặt đúng phiên bản tương thích với CUDA 12.x của H100 để vLLM hoạt động trơn tru.

## 3. Quy trình Chạy (Execution)

### Bước 1: Nạp Dữ liệu (Ingestion)
Đây là công đoạn nặng nhất (500k tài liệu).
```bash
cd TRAG
python ingest.py
```
*Lưu ý:* Hàm `model.encode()` trong file này sẽ dùng sức mạnh của H100 để embed các text. Quá trình này có thể tốn một khoảng thời gian tuỳ vào cấu hình I/O của server. Dữ liệu sau khi nạp sẽ nằm ở thư mục `lancedb_data/`.

### Bước 2: Chạy Benchmark Pipeline
Sau khi database đã sẵn sàng:
```bash
python run_benchmark.py
```
Luồng này sẽ chiếm gần như toàn bộ VRAM để load mô hình Llama-3-8B và mô hình Reranker. Tại mỗi `Stage`, bạn sẽ thấy GPU H100 load batch hàng trăm query/document cùng lúc. Kết quả cuối cùng sẽ được lưu tại `EnterpriseRAG-Bench/answers.jsonl`.

### Bước 3: Đánh giá bằng Script của Repo
Dùng tool có sẵn của Onyx để chấm điểm Leaderboard:
```bash
cd ../EnterpriseRAG-Bench
python -m src.scripts.answer_evaluation.metrics_based_eval --answers-file answers.jsonl
```

## 4. Tùy chỉnh thêm (Mở rộng)
> [!NOTE]
> Do bộ dataset chuẩn từ file `.txt` không chứa trường Timestamp cụ thể, hàm Time Decay hiện tại đang set `lambda_val = 0` (không phạt) để chạy benchmark gốc. Nếu bạn có một dataset JSON bổ sung chứa Metadata `Date` (như nhắc tới trong `extra_questions`), bạn chỉ cần extract field đó trong `ingest.py` và đưa vào công thức $e^{-\lambda \Delta t}$ ở Stage 3 của `trag_pipeline.py`.
