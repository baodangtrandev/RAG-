# Pipeline Đánh giá Benchmark Toàn diện (T-RAG vs Baselines) bằng Local LLM-as-a-Judge

Dựa trên bài báo **EnterpriseRAG-Bench** và yêu cầu của bạn, chúng ta sẽ xây dựng một quy trình benchmarking toàn diện. Quy trình này không chỉ đánh giá các siêu tham số (hyperparameters) trong cấu hình T-RAG hiện tại của bạn, mà còn so sánh nó với các pipeline cơ bản (baselines) được đề cập trong bài báo (như BM25, Vector Search). 

Vì không có API của OpenAI, chúng ta sẽ sử dụng một LLM Local qua `vLLM` để đóng vai trò làm LLM-Judge. Biến môi trường `JUDGE_LLM_MODEL` sẽ được định nghĩa trong `.env` để bạn có thể linh hoạt thay đổi model (khuyến nghị các model <= 14B để vừa với 40GB VRAM).

## Quy trình Đề xuất (4 Bước)

### Phase 1: Triển khai & Chạy Baseline Pipelines (Từ Bài Báo)
Bài báo sử dụng 3 baselines: BM25, Vector Search, và Bash Agent. Chúng ta sẽ implement 2 baseline phổ biến và khả thi nhất để chạy tự động:
1. **BM25 Pipeline:** Sử dụng BM25 (qua thư viện như `rank_bm25` hoặc ElasticSearch/OpenSearch) để truy xuất top-10 tài liệu, sau đó đưa vào LLM để tạo câu trả lời.
2. **Vector Search Pipeline:** Sử dụng mô hình embedding chuẩn (ví dụ `BAAI/bge-large-en-v1.5`) và vector database (LanceDB) để lấy top-10 tài liệu, sau đó sinh câu trả lời.

*Đầu ra của Phase 1:* `results/baseline_bm25.jsonl`, `results/baseline_vector.jsonl`.

### Phase 2: Chạy T-RAG Pipeline với các tham số (Ablation Study)
Đây là bước chạy script `run_benchmark.py` của bạn với nhiều tổ hợp cấu hình (config) khác nhau để tìm ra bộ tham số tối ưu (Hyperparameter Tuning). Một số kịch bản chạy (có thể định nghĩa qua file shell script `run_all.sh`):

1. **Test tính năng CSEP:** `ENABLE_CSEP_FOR_ALL="True"` vs `"False"`
2. **Test ngưỡng Reranker:** `RERANKER_THRESHOLD="0.0"` (Không lọc) vs `RERANKER_THRESHOLD="0.5"`
3. **Test thông số lấy tài liệu:** `RAG_TOP_K_RETRIEVE=20` / `RAG_TOP_K_FINAL=5` vs `RAG_TOP_K_RETRIEVE=50` / `RAG_TOP_K_FINAL=10`
4. **Test Router Tau:** `RAG_TAU="0.15"` vs `RAG_TAU="0.5"`

*Đầu ra của Phase 2:* Các file `.jsonl` tương ứng như `results/trag_csep_true.jsonl`, `results/trag_tau_0.5.jsonl`, v.v.

### Phase 3: Khởi động Local API Server cho LLM Judge
Sử dụng `vLLM` để bật một API Server chuẩn OpenAI, phục vụ cho việc chấm điểm. Model sẽ được đọc từ file `.env` của bạn (ví dụ thông qua biến `JUDGE_LLM_MODEL`).

```bash
# Đọc tên model từ env và khởi chạy vLLM (chạy trên một terminal/pane khác)
source .env
vllm serve $JUDGE_LLM_MODEL --port 8000 --max-model-len 8192
```

### Phase 4: Patch mã nguồn & Chạy Đánh giá tự động (Evaluation)
Bài báo cung cấp script `src.scripts.answer_evaluation.metrics_based_eval`. Tôi sẽ:
1. **Patch mã nguồn Evaluation:** Tìm và sửa các file gọi OpenAI API trong source của Benchmark để trỏ `base_url` về `http://localhost:8000/v1` và truyền tên model động từ `JUDGE_LLM_MODEL`.
2. **Chạy Script Đánh Giá:** Viết một script tổng hợp để lặp qua tất cả các file `results/*.jsonl` (từ Phase 1 và 2) và gọi lệnh đánh giá. LLM Judge sẽ bầu chọn 3 lần (three-judge consensus) để tính điểm Correctness và Completeness.
3. **Tổng hợp Báo Cáo:** Xuất ra một bảng tóm tắt so sánh điểm số giữa BM25, Vector Search và các cấu hình T-RAG của bạn.

---

## User Review Required

> [!IMPORTANT]
> - **Cài đặt Baselines:** Bạn có muốn tôi viết mã nguồn cho 2 pipeline cơ bản (BM25 và Vector Search) bằng Python luôn không, hay bạn đã có sẵn mã nguồn của bài báo?
> - **Cập nhật `.env`:** Tôi sẽ thêm biến `JUDGE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"` (hoặc model bạn muốn) vào file `.env` của bạn. Bạn có đồng ý không?
> - **Nguồn mã đánh giá:** Thư mục `src/scripts/answer_evaluation/` (của Benchmark) hiện đã tồn tại trong dự án của bạn chưa? Nếu chưa, tôi sẽ cần tạo cấu trúc thư mục và tải các đoạn script đánh giá tương ứng về.
