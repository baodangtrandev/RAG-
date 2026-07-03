# Phân tích Khoảng trống Nghiên cứu (Research Gap Analysis)

## 1. Đặt vấn đề (Problem Statement)
Các hệ thống Retrieval-Augmented Generation (RAG) đã đạt được những thành công đáng kể trên các cơ sở tri thức sạch, có cấu trúc từ điển bách khoa (ví dụ: Wikipedia). Tuy nhiên, khi được triển khai trong môi trường doanh nghiệp quy mô lớn (large-scale enterprise datasets) như EnterpriseRAG-Bench với hơn 500.000 tài liệu đa dạng (Slack, Gmail, Jira, Confluence), kiến trúc "Standard RAG" (chỉ sử dụng Vector Search kết hợp LLM) bộc lộ sự suy giảm hiệu suất nghiêm trọng. 

## 2. Các khoảng trống nghiên cứu đã xác định (Identified Gaps)

### Điểm yếu 1: Sự mù mờ về thời gian (Temporal Blindness)
- **Triệu chứng:** Standard RAG thường xuyên trả về thông tin đã lỗi thời cho những truy vấn nhạy cảm về thời gian (time-sensitive queries). Ví dụ: Với câu hỏi "Quy trình xin nghỉ phép hiện tại là gì?", hệ thống có thể trả về chính sách của năm 2022 thay vì 2024.
- **Nguyên nhân gốc rễ:** Độ tương đồng Cosine (Cosine similarity) trong không gian Vector thuần túy chỉ đo lường khoảng cách về ngữ nghĩa (semantic distance) mà không hề có nhận thức về siêu dữ liệu thời gian (timestamp metadata). Một tài liệu cũ nhưng có nội dung trùng khớp từ vựng cao vẫn sẽ đánh bại một tài liệu mới cập nhật.
- **Giải pháp của T-RAG:** Áp dụng **Xếp hạng lại theo suy giảm thời gian có điều kiện (Conditional Temporal Reranking)**. Điểm số cuối cùng sẽ là sự kết hợp giữa Điểm liên quan (Relevance Score) và Hàm suy giảm thời gian hàm mũ ($e^{-\lambda \Delta t}$). Việc sử dụng cờ điều kiện (`requires_latest`) đảm bảo rằng các truy vấn tra cứu sự kiện lịch sử sẽ không bị ảnh hưởng bởi hàm suy giảm này.

### Điểm yếu 2: Mật độ Không gian Vector khi mở rộng (Vector Space Density at Scale)
- **Triệu chứng:** Độ chính xác (Precision) giảm mạnh khi số lượng tài liệu vượt mức 500.000. Rất nhiều tài liệu trả về tuy có chung chủ đề ngữ nghĩa nhưng lại thuộc sai ngữ cảnh hoặc sai dự án.
- **Nguyên nhân gốc rễ:** Khi không gian vector chứa hàng trăm ngàn tài liệu hội thoại ngắn (như 285.000 tin nhắn Slack), khoảng cách cosine giữa các vector trở nên quá gần nhau và kém phân biệt (dense clustering). Thuật toán tìm kiếm xấp xỉ (ANN) gặp khó khăn trong việc tìm ra chính xác kết quả đúng giữa một "rừng" các kết quả na ná nhau.
- **Giải pháp của T-RAG:** Đưa vào **Tiền lọc Siêu dữ liệu có Định hướng (Targeted Metadata Pre-filtering)**. Bằng cách sử dụng LLM để phân tích và trích xuất ý định tìm kiếm nguồn từ truy vấn (ví dụ: "chỉ tìm trong Jira"), ta sử dụng bộ lọc SQL/Relational trong LanceDB để thu hẹp không gian tìm kiếm trước khi chạy thuật toán ANN. Điều này giúp loại bỏ hoàn toàn nhiễu ngữ nghĩa (semantic noise) từ các nguồn không liên quan và giảm độ trễ (latency).

### Điểm yếu 3: Bất đồng từ vựng trong văn cảnh doanh nghiệp (Vocabulary Mismatch)
- **Triệu chứng:** Người dùng thường tìm kiếm bằng ngôn ngữ tự nhiên (ví dụ: "lỗi không kết nối được database hôm qua"), nhưng tài liệu nội bộ (như Jira, GitHub) lại chứa các mã định danh, ID ticket, hoặc thuật ngữ kỹ thuật cứng (ví dụ: "Error 503 on DB-01", "Ticket PROJ-102"). Standard RAG thường bỏ sót các kết quả quan trọng này.
- **Nguyên nhân gốc rễ:** Các mô hình nhúng Vector (như BGE-large) được huấn luyện để bắt "ý nghĩa tổng quát" nhưng lại tỏ ra cực kỳ kém hiệu quả trong việc đối khớp chính xác (exact lexical match) các mã số, tên riêng, hoặc các từ ngoài từ điển (out-of-vocabulary terms).
- **Giải pháp của T-RAG:** Triển khai **Truy xuất Lai kết hợp Tự động Mở rộng Truy vấn (Hybrid Retrieval + Self-Query Expansion)**. Đầu tiên, truy vấn được LLM tự động mở rộng (bổ sung từ đồng nghĩa và tạo tài liệu giả định - HyDE). Sau đó, quá trình truy xuất sẽ chạy song song cả Dense Search (Vector) và Sparse Search (BM25 - Tantivy), cuối cùng gộp lại thông qua thuật toán Reciprocal Rank Fusion (RRF). Nhờ mảng BM25, hệ thống sẽ không bao giờ bỏ sót các từ khóa định danh quan trọng.

---

## 3. Các Đóng góp của T-RAG (Contributions)
Để giải quyết triệt để các khoảng trống nghiên cứu trên, kiến trúc T-RAG mang đến 4 đóng góp kỹ thuật cốt lõi:
1. **Lưu trữ hợp nhất (Unified Storage với LanceDB):** Thiết lập một cơ sở dữ liệu duy nhất hỗ trợ đồng thời tìm kiếm Vector, tìm kiếm toàn văn bản (FTS), và lọc Metadata, loại bỏ độ trễ và sự phân mảnh của việc dùng nhiều cơ sở dữ liệu cùng lúc.
2. **Tự động mở rộng truy vấn (Self-Query Expansion):** Sử dụng bộ phân tích dựa trên LLM để hiểu sâu ngữ cảnh, phát hiện yếu tố thời gian (temporal intent), và mở rộng truy vấn (bao gồm kỹ thuật HyDE) trước khi tiến hành tìm kiếm.
3. **Truy xuất lai (Hybrid Retrieval):** Tích hợp hoàn hảo sức mạnh ngữ nghĩa của Vector Search và sức mạnh đối khớp từ khóa của BM25, dung hòa kết quả thông qua thuật toán RRF.
4. **Xếp hạng lại theo thời gian có điều kiện (Conditional Temporal Reranking):** Sử dụng mô hình Cross-Encoder kết hợp hàm Time Decay để tinh chỉnh top-K kết quả cuối cùng, đảm bảo thông tin trả về luôn phản ánh thực trạng mới nhất và chính xác nhất của doanh nghiệp.
