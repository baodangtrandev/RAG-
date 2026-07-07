# Literature Review cho T-RAG (Targeted RAG)

Tài liệu này tổng hợp các bài báo quan trọng nhất làm nền tảng cho việc thiết kế kiến trúc T-RAG (Probabilistic Source-Aware RAG), giải quyết thách thức trên tập dữ liệu EnterpriseRAG-Bench.

---

## 1. Nền tảng về RAG (Retrieval-Augmented Generation)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 1 | **Lewis et al. (2020)**<br>*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Giới thiệu mô hình RAG đầu tiên kết hợp pre-trained seq2seq (BART) với non-parametric memory. | Standard RAG Baseline. |
| 2 | **Gao et al. (2023)**<br>*Retrieval-Augmented Generation for Large Language Models: A Survey* | Phân loại RAG thành Naive, Advanced và Modular RAG. | Cung cấp taxonomy định vị T-RAG như một Modular RAG. |

---

## 2. Định tuyến Ngữ nghĩa (Semantic Routing & Pre-Retrieval)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 3 | **Qiao et al. (2024)**<br>*Route Before Retrieve: A New Paradigm for Retrieval-Augmented Generation* | Đề xuất "Route Before Retrieve", sử dụng LLM để xác định nguồn/hành động trước khi tìm kiếm. | Khái niệm nền tảng cho module **Probabilistic Source Router (PSR)**. |
| 4 | **Wang et al. (2024)**<br>*RAGRouter: Learning to Route in Retrieval-Augmented Generation* | Huấn luyện mô hình Router chuyên dụng để định tuyến câu hỏi tới đúng index dữ liệu. | Hỗ trợ cơ sở lý thuyết huấn luyện Classifier cho PSR. |

---

## 3. Truy xuất Lai (Hybrid Retrieval & Fusion)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 5 | **Robertson et al. (2009)**<br>*The Probabilistic Relevance Framework: BM25 and Beyond* | Trình bày toán học thuật toán BM25 (TF-IDF cải tiến). | Giải quyết "Vocabulary Mismatch" kết hợp cùng Dense Search. |
| 6 | **Cormack et al. (2009)**<br>*Reciprocal Rank Fusion (RRF) Outperforms Data Fusion Algorithms* | Đề xuất RRF để kết hợp kết quả từ nhiều bộ tìm kiếm: `1 / (k + rank)`. | Thành phần cốt lõi được T-RAG cải tiến thành **Source-Weighted RRF (SW-RRF)**. |

---

## 4. Suy luận Đa bước (Multi-hop Reasoning)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 7 | **Tang et al. (2024)**<br>*MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries* | Đánh giá khả năng tìm kiếm thông tin chuỗi trên tài liệu rời rạc. | Động lực để phát triển module **Cross-Source Entity Propagation (CSEP)**. |
| 8 | **Zhang et al. (2025)**<br>*HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation* | Cải thiện luồng logic tìm kiếm đa bước. | Mô phỏng cơ sở lý thuyết đồ thị hai phía trong thuật toán CSEP. |

---

## 5. Đánh giá độ tin cậy (Cross-Encoder Reranking)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 9 | **Nogueira & Cho (2019)**<br>*Passage Re-ranking with BERT* | Lần đầu tiên sử dụng Cross-Encoder (BERT) để rerank lại top-K kết quả. | Cơ sở cho module Reranker với Hard Thresholding lọc nhiễu. |

---

## Kết luận (Literature Synthesis)

Thông qua phân tích tài liệu, hệ thống **T-RAG** giải quyết bài toán cốt lõi bằng cách:
- Áp dụng triệt để tư tưởng [3], [4] để thiết kế **Database Sharding + PSR**, giải quyết bài toán chi phí (Scalability).
- Cải tiến thuật toán RRF [6] thành **SW-RRF** với Bayesian Prior nhằm khắc phục giới hạn của truy xuất lai truyền thống.
- Vận dụng mô hình [7], [8] vào bài toán truy xuất đa nguồn (Cross-Source) trong doanh nghiệp thông qua **CSEP**.
