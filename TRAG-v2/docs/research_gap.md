# Phân tích Khoảng trống Nghiên cứu (Research Gap Analysis)
# **Cập nhật sau khi EDA dataset thực tế**

---

## 1. Đặc điểm thực tế của Dataset (Revised after EDA)

**Kết quả EDA từ `data/documents/test.parquet`:**
- **Tổng số tài liệu:** 511,962 documents
- **Cột dữ liệu:** `doc_id`, `source_type`, `title`, `content`
- **Không có timestamp:** Dataset **không chứa thông tin thời gian**
- **Không có missing values**
- **Độ dài tài liệu:** Trung bình ~667 từ, max 6,230 từ, std ~327 từ (phân bố lệch phải)

> [!WARNING]
> Đây là phát hiện quan trọng: Bài báo gốc mô tả hệ thống xử lý dữ liệu "có tần suất cập nhật cao" nhưng bộ dataset benchmark thực tế lại **không có timestamp**. Điều này yêu cầu điều chỉnh lại thiết kế hệ thống T-RAG.

---

## 2. Đặt vấn đề (Problem Statement)

Các hệ thống Retrieval-Augmented Generation (RAG) đạt thành công lớn trên dữ liệu sạch kiểu Wikipedia, nhưng gặp khó khăn trên tập dữ liệu doanh nghiệp quy mô lớn như **EnterpriseRAG-Bench** với hơn **511,962 tài liệu** từ nhiều nguồn không đồng nhất. Kiến trúc "Standard RAG" bộc lộ sự suy giảm hiệu suất do **hai vấn đề cốt lõi vẫn còn nguyên giá trị** sau khi EDA.

---

## 3. Các khoảng trống nghiên cứu đã xác định (Revised Gaps)

### Điểm yếu 1: Mật độ Không gian Vector & Sự nhập nhằng nguồn (Vector Density + Source Ambiguity)
- **Triệu chứng:** Với 511,962 documents phủ qua nhiều nguồn (Slack, Gmail, Jira, Confluence, GitHub...), độ tương đồng Cosine kém phân biệt. Tài liệu từ sai nguồn nhưng cùng chủ đề thường bị truy xuất nhầm.
- **Ví dụ:** Truy vấn "deployment status" có thể trả về cả Slack message, Confluence runbook, và Jira ticket — nhưng người dùng chỉ cần loại cụ thể.
- **Nguyên nhân gốc rễ:** Standard RAG tìm kiếm trên toàn bộ vector space mà không phân biệt nguồn tài liệu (source_type).
- **Giải pháp T-RAG:** **Targeted Metadata Pre-filtering** — Dùng LLM phân tích truy vấn để trích xuất `source_filter` (ví dụ: "chỉ tìm trong Confluence"), sau đó lọc SQL theo `source_type` trong LanceDB **trước** khi chạy ANN. Thu hẹp không gian tìm kiếm từ 511k → vài chục nghìn docs.

### Điểm yếu 2: Bất đồng từ vựng trong văn cảnh doanh nghiệp (Vocabulary Mismatch)
- **Triệu chứng:** Người dùng hỏi bằng ngôn ngữ tự nhiên ("lỗi kết nối database"), tài liệu chứa ID, mã kỹ thuật ("Error 503", "PR #482", "ticket PROJ-102").
- **Ví dụ:** Truy vấn "meeting about GPU budget" nhưng Fireflies transcript dùng "H100 compute allocation Q3".
- **Nguyên nhân gốc rễ:** Dense Embedding tốt về ngữ nghĩa nhưng kém khớp exact term.
- **Giải pháp T-RAG:** **Hybrid Retrieval (Dense + BM25) + Self-Query Expansion** — Dense Search bắt ngữ nghĩa, BM25 bắt exact term, RRF gộp cả hai; HyDE mở rộng truy vấn để tăng recall.

### Điểm yếu 3: Suy giảm hiệu suất ở Multi-Source Retrieval (Cross-Source Context)
- **Triệu chứng:** Câu hỏi phức tạp yêu cầu tổng hợp thông tin từ nhiều nguồn (ví dụ: "AI task mentioned in Slack có được implement trong GitHub chưa?") rất khó trả lời đúng với Standard RAG.
- **Nguyên nhân:** Standard RAG chỉ retrieve đơn giản top-K, không nhận biết mối liên hệ cross-source.
- **Giải pháp T-RAG:** **Multi-pass Retrieval** — Lần 1 retrieve tài liệu chính, lần 2 retrieve tài liệu liên quan từ nguồn khác dựa trên context đã tìm thấy.

---

## 4. Điều chỉnh kiến trúc T-RAG sau EDA

| Component | Kế hoạch ban đầu | Điều chỉnh thực tế |
|-----------|-----------------|-------------------|
| Temporal Reranker | Time Decay: `Score × e^(−λΔt)` | ❌ **Loại bỏ** — Dataset không có timestamp. Thay bằng **Cross-Encoder Reranker thuần** (BGE-Reranker) |
| Metadata Filtering | Lọc theo `source_type` + `timestamp` | ✅ **Giữ nguyên** — Lọc theo `source_type` vẫn có giá trị lớn |
| Hybrid Search | Dense + BM25 với RRF | ✅ **Giữ nguyên và nâng tầm quan trọng** |
| Self-Query Expansion | Detect temporal intent + HyDE | ✅ **Giữ phần Source Detection + HyDE**, bỏ temporal intent detection |
| Contribution | "Temporal & Targeted" RAG | **Đổi tên thành "Targeted RAG"** — Nhấn mạnh vào Source-Aware Retrieval |

---

## 5. Đóng góp đã cập nhật của T-RAG (Revised Contributions)

1. **Lưu trữ hợp nhất (Unified Storage với LanceDB):** Một DB duy nhất cho Vector + FTS + SQL Metadata.
2. **Tự động nhận diện nguồn và mở rộng truy vấn (Source-Aware Self-Query Expansion):** LLM phân tích truy vấn để xác định `source_type` phù hợp và sinh truy vấn mở rộng (HyDE).
3. **Truy xuất lai theo nguồn (Source-Filtered Hybrid Retrieval):** Áp dụng Metadata Pre-filter → Dense + Sparse (BM25) → RRF fusion trong không gian tìm kiếm đã được thu hẹp.
4. **Cross-Encoder Reranking:** BGE-Reranker tinh chỉnh top-50 → top-10 bằng query-document cross-attention để đảm bảo precision tối đa.
