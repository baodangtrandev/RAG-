# Phân tích Khoảng trống Nghiên cứu (Research Gap Analysis)
# **Cập nhật theo Proposal T-RAG (Probabilistic Source-Aware RAG)**

---

## 1. Đặt vấn đề (Problem Statement)

Các hệ thống Retrieval-Augmented Generation (RAG) đạt thành công lớn trên dữ liệu được làm sạch như Wikipedia, nhưng gặp khó khăn khi triển khai trên tập dữ liệu doanh nghiệp quy mô lớn như **EnterpriseRAG-Bench** với hơn **511,962 tài liệu** từ nhiều nguồn không đồng nhất. Kiến trúc "Standard RAG" bộc lộ sự suy giảm hiệu suất và lãng phí chi phí tính toán nghiêm trọng do những điểm yếu cốt lõi về hệ thống và thuật toán.

---

## 2. Các khoảng trống nghiên cứu đã xác định (Research Gaps)

### Điểm yếu 1: Mật độ Không gian Vector & Sự nhập nhằng nguồn (Vector Density + Source Ambiguity)
- **Triệu chứng:** Với 511,962 documents phủ qua 9 nguồn (Slack, Gmail, Jira, Confluence, GitHub...), độ tương đồng Cosine kém phân biệt. Rất nhiều tài liệu từ sai nguồn nhưng có cùng từ khóa hoặc chủ đề sẽ bị truy xuất nhầm, trong khi chi phí quét toàn bộ index (I/O, FLOPs) là rất cao.
- **Nguyên nhân gốc rễ:** Standard RAG tìm kiếm trên toàn cục không gian vector (Brute-force/Toàn bộ đồ thị ANN) mà không định hướng nguồn dữ liệu.
- **Giải pháp T-RAG (PSR & DB Sharding):** Áp dụng mô hình **Database Sharding** phân chia dữ liệu thành $N$ bảng vật lý độc lập. Xây dựng **Probabilistic Source Router (PSR)**, một classifier dự đoán xác suất nguồn $P(s_i|Q)$ để "định tuyến" câu hỏi. Chỉ index của các nguồn có xác suất cao mới được load vào bộ nhớ, giảm Search Space từ 50-90%.

### Điểm yếu 2: Hạn chế của Truy xuất Lai truyền thống (Naive Hybrid Fusion)
- **Triệu chứng:** Khi dùng Hybrid Search (Vector Dense + BM25 Sparse), thuật toán kết hợp truyền thống là Reciprocal Rank Fusion (RRF) sẽ đánh đồng mọi tài liệu, làm mất đi tính "ưu tiên" cho đúng nguồn.
- **Nguyên nhân gốc rễ:** Thuật toán RRF gốc không xem xét xác suất (Prior Knowledge) của việc tài liệu thuộc về nguồn nào.
- **Giải pháp T-RAG (SW-RRF):** Đề xuất **Source-Weighted Reciprocal Rank Fusion (SW-RRF)**, một cải tiến toán học bằng cách đưa Bayesian Prior $P(s_d|Q)$ làm trọng số nhân trực tiếp vào công thức RRF, ép các tài liệu đúng nguồn được xếp hạng cao hơn tài liệu đồng nghĩa nhưng sai nguồn.

### Điểm yếu 3: Suy giảm hiệu suất ở Truy xuất Đa bước (Multi-Source Queries)
- **Triệu chứng:** Các câu hỏi phức tạp yêu cầu tổng hợp thông tin chéo nguồn (ví dụ: "Tính năng ABC thảo luận ở Slack đã được xử lý trong Jira chưa?") vượt quá khả năng của RAG đơn bước.
- **Nguyên nhân gốc rễ:** Standard RAG chỉ retrieve một lần trên một cụm thông tin rời rạc.
- **Giải pháp T-RAG (CSEP):** Triển khai thuật toán **Cross-Source Entity Propagation (CSEP)**, mô phỏng đồ thị hai phía (Bipartite Graph). RAG sẽ thực hiện Hop 1 trên nguồn có xác suất cao nhất, trích xuất thực thể mỏ neo (Anchor Entities), sau đó dùng thực thể này làm truy vấn mở rộng cho Hop 2 trên nguồn còn lại.

---

## 3. Kiến trúc Đánh giá Mới (New Evaluation Framework)

Để chứng minh luận điểm, hệ thống T-RAG sẽ không chỉ đánh giá về chất lượng mà còn về hiệu năng hệ thống:
1. **Chất lượng:** Đo lường Recall@10, NDCG@10 trên 500 truy vấn của tập test.
2. **Hiệu năng Hệ thống (System Metrics):** Đo lường sự giảm thiểu về Search Space (tính bằng % số lượng documents cần tính toán) và Latency thực tế (ms) so với Standard RAG.
3. **Độ tin cậy:** Khả năng từ chối trả lời (Unanswerable) giảm hallucination qua module Hard Thresholding Reranker.
