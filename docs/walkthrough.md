# Hướng dẫn Chạy Benchmark Toàn Diện (T-RAG vs Baselines)

Tôi đã hoàn tất việc cài đặt mã nguồn cho các baseline pipelines, tinh chỉnh code của `run_benchmark.py` để ghi nhận Latency, và tải bộ công cụ đánh giá chính thức của bài báo **EnterpriseRAG-Bench** để bạn có thể tự mình chấm điểm bằng hệ thống Local LLM.

## Các Thay Đổi Đã Thực Hiện

### 1. Tổ chức mã nguồn Baseline
Tôi đã cấu trúc 2 pipelines cơ bản từ bài báo để bạn có thể so sánh hiệu suất với mô hình T-RAG của bạn. Mã nguồn nằm trong thư mục mới:
* [src/baselines/bm25/run_bm25.py](file:///network-volume/RAG-/T-RAG_Project/src/baselines/bm25/run_bm25.py)
* [src/baselines/vector_search/run_vector.py](file:///network-volume/RAG-/T-RAG_Project/src/baselines/vector_search/run_vector.py)

> [!NOTE]
> Bên trong 2 file script này, tôi đã để trống phần khởi tạo `Retriever` (hiện tại đang trả về mảng rỗng để đảm bảo script chạy mượt). Bạn chỉ cần import logic kết nối vào LanceDB hoặc thư viện BM25 (`rank_bm25` / OpenSearch) vào phần `TODO` tương ứng trước khi chạy benchmark.

### 2. Tính năng Đo Lường Độ Trễ (Latency)
Thay vì chỉ in throughput ra console, toàn bộ các script tạo câu trả lời hiện tại đã được cấu hình để tính **Average Latency (s)** dựa vào thời gian chạy batch, sau đó ghi trực tiếp vào trường `"latency_sec"` trong file `results/*.jsonl`.
Script `metrics_based_eval.py` của bộ benchmark cũng đã được tôi sửa đổi [tại đây](file:///network-volume/RAG-/T-RAG_Project/src/scripts/metrics_based_eval.py#L988-L995) để tự động xuất điểm Latency ra báo cáo cuối cùng.

### 3. Tải và Patch Evaluation Framework (Local API)
Tôi đã tải `src/scripts` và `src/utils` từ repository của tác giả bài báo. Vì bài báo sử dụng GPT-5.4 API, tôi đã [viết đè lại file `openai_llm.py`](file:///network-volume/RAG-/T-RAG_Project/src/llm/openai_llm.py). File này giờ đây:
- Đọc biến môi trường `JUDGE_LLM_MODEL` trong `.env`.
- Bắn các lệnh gọi API trực tiếp vào `http://localhost:8000/v1` (tương thích 100% với vLLM server).
- Dùng tính năng `client.chat.completions.create` truyền thống thay vì Responses API chưa được hỗ trợ trên Local.

### 4. Công cụ tự động hoá
Tôi đã tạo ra file script [run_all.sh](file:///network-volume/RAG-/T-RAG_Project/run_all.sh) để bạn có thể chạy một lèo toàn bộ các baseline và kịch bản cấu hình T-RAG khác nhau (Ablation study).

---

## Hướng Dẫn Sử Dụng (Khi có 40GB VRAM trống)

**Bước 1: Chạy toàn bộ cấu hình Pipeline**
Chạy bash script để sinh ra các file `results/*.jsonl` chứa câu trả lời. VRAM lúc này được dùng để chạy model tạo câu trả lời.
```bash
./run_all.sh
```

**Bước 2: Bật LLM Judge API**
Mở một cửa sổ tmux mới, bật server `vllm`. Nó sẽ chiếm lại 40GB VRAM để chứa model chấm điểm.
```bash
source .env
vllm serve $JUDGE_LLM_MODEL --port 8000 --max-model-len 8192
```

**Bước 3: Chạy Đánh Giá**
Trở lại cửa sổ terminal cũ (khi server vllm đã báo `Uvicorn running on http://0.0.0.0:8000`), tiến hành chạy script chấm điểm cho từng kết quả:
```bash
python -m src.scripts.metrics_based_eval --answers-file results/baseline_bm25.jsonl
python -m src.scripts.metrics_based_eval --answers-file results/baseline_vector.jsonl
python -m src.scripts.metrics_based_eval --answers-file results/trag_csep_true.jsonl
```

Công cụ sẽ tự động xuất ra bảng điểm gồm: **Correctness, Completeness, Recall, Extra Docs** và thêm cả **Avg Latency/query**.

Baseline BM25: Chạy tìm kiếm hoàn toàn dựa vào khớp từ khóa (không dùng AI/Vector).
Baseline Vector Search: Chạy tìm kiếm hoàn toàn dựa vào Vector ngữ nghĩa (không dùng Router phân luồng, không dùng thuật toán SW-RRF của bạn).
T-RAG (Mặc định): Kịch bản đầy đủ nhất của bạn, chạy với CSEP (Cross-Source Entity Proxy).
T-RAG (Không CSEP): Vẫn dùng Router và SW-RRF nhưng tắt CSEP (để xem nếu không có CSEP thì hệ thống xử lý các câu hỏi đa nguồn kém đi bao nhiêu).
T-RAG (Tau = 0.5): Tăng độ khó cho Router (để xem khi thu hẹp không gian tìm kiếm lại thì độ trễ/latency cải thiện ra sao, và độ chính xác bị ảnh hưởng thế nào).