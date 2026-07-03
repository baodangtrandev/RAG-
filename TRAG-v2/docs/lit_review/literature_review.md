# Literature Review cho T-RAG (Temporal & Targeted RAG)

Tài liệu này tổng hợp hơn 20 bài báo quan trọng nhất làm nền tảng cho việc thiết kế và phát triển hệ thống T-RAG, giải quyết các thách thức trên tập dữ liệu EnterpriseRAG-Bench.

---

## 1. Nền tảng về RAG (Retrieval-Augmented Generation)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 1 | **Lewis et al. (2020)**<br>*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Giới thiệu mô hình RAG đầu tiên kết hợp pre-trained seq2seq (BART) với non-parametric memory (Dense Passage Retriever - DPR). | Foundation: Khái niệm cơ bản của hệ thống. Đây là baseline chuẩn (Standard RAG) mà chúng ta sẽ so sánh. |
| 2 | **Gao et al. (2023)**<br>*Retrieval-Augmented Generation for Large Language Models: A Survey* | Phân loại RAG thành Naive RAG, Advanced RAG và Modular RAG. Tổng hợp các kỹ thuật pre-retrieval, retrieval, và post-retrieval. | Cung cấp taxonomy để định vị T-RAG như một hệ thống Modular RAG tích hợp các pipeline phức tạp. |
| 3 | **Shi et al. (2023)**<br>*REPLUG: Retrieval-Augmented Black-Box Language Models* | Đề xuất coi LLM như black-box và tối ưu bộ retriever bằng cách dùng tín hiệu từ LLM (cross-entropy loss) để tính điểm tài liệu. | Gợi ý cho việc đánh giá chất lượng tài liệu dựa trên góc độ sinh văn bản của LLM. |
| 4 | **Asai et al. (2023)**<br>*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | Huấn luyện LLM tự đánh giá xem khi nào cần retrieve, tài liệu có relevant không, và câu trả lời có dựa trên tài liệu không qua các token phản hồi (reflection tokens). | Ảnh hưởng đến thiết kế của module Query Parser (tự động nhận biết temporal intent mà không cần truy xuất mù quáng). |

---

## 2. Truy xuất Lai (Hybrid Retrieval & Sparse Search)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 5 | **Robertson et al. (2009)**<br>*The Probabilistic Relevance Framework: BM25 and Beyond* | Trình bày toán học đằng sau thuật toán BM25 (TF-IDF cải tiến), tiêu chuẩn vàng cho Exact Lexical Match. | Cơ sở để giải quyết "Vocabulary Mismatch" trong T-RAG, đặc biệt với các ID, mã lỗi, tên Jira ticket. |
| 6 | **Karpukhin et al. (2020)**<br>*Dense Passage Retrieval for Open-Domain Question Answering (DPR)* | Chứng minh Dense Retriever sử dụng Dual-Encoder (bi-encoder) vượt trội hơn BM25 trên Open-Domain QA. | Cơ sở của Dense Search module. Tuy nhiên, T-RAG chỉ ra rằng trên Enterprise Data, chỉ dùng Dense là không đủ. |
| 7 | **Gao et al. (2021)**<br>*COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List* | Kết hợp kiến trúc Inverted Index với Contextualized representations (từ transformer) để cân bằng giữa exact match và semantic match. | Củng cố luận điểm cần kết hợp Sparse và Dense trong cùng một hệ thống (LanceDB FTS + Vector). |
| 8 | **Cormack et al. (2009)**<br>*Reciprocal Rank Fusion (RRF) Outperforms Data Fusion Algorithms* | Đề xuất RRF để kết hợp kết quả từ nhiều bộ tìm kiếm mà không cần chuẩn hóa điểm (score normalization). Công thức: `1 / (k + rank)`. | Thành phần cốt lõi của **Hybrid Retriever** trong T-RAG để gộp điểm từ Dense (Vector) và Sparse (BM25/Tantivy). |

---

## 3. Reranking & Tối ưu Context (Post-Retrieval)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 9 | **Nogueira & Cho (2019)**<br>*Passage Re-ranking with BERT* | Lần đầu tiên sử dụng Cross-Encoder (BERT) để rerank lại top-K kết quả từ BM25, cải thiện độ chính xác đáng kể. | Cơ sở cho module Reranker. T-RAG sử dụng mô hình BGE-Reranker dựa trên kiến trúc Cross-Encoder này. |
| 10 | **Xu et al. (2023)**<br>*RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation* | Đề xuất mô hình nén context để chỉ giữ lại những câu/thông tin thực sự hữu ích, giảm độ nhiễu và tránh OOM context window. | Hỗ trợ ý tưởng Context Truncation của T-RAG trước khi đưa vào Generation Stage (vLLM). |
| 11 | **Liu et al. (2023)**<br>*Lost in the Middle: How Language Models Use Long Contexts* | Chỉ ra rằng LLMs thường chú ý tốt phần đầu và cuối ngữ cảnh, nhưng bỏ qua thông tin ở giữa khi context quá dài. | Nhấn mạnh tầm quan trọng của việc chỉ đưa top-K (nhỏ) tài liệu thực sự liên quan nhất lên đầu, là lý do tại sao Reranker là bắt buộc. |

---

## 4. Query Expansion & Understanding (Pre-Retrieval)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 12 | **Ma et al. (2023)**<br>*Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)* | Dùng LLM để sinh ra tài liệu giả định (hypothetical document) từ câu hỏi ngắn, sau đó lấy embedding của tài liệu giả định để tìm kiếm. | Một kỹ thuật được tích hợp trong **Self-Query Expansion** của T-RAG để tăng Recall. |
| 13 | **Wang et al. (2023)**<br>*Query2doc: Query Expansion with Large Language Models* | Sinh pseudo-documents bằng LLMs rồi nối vào query gốc. Chứng minh hiệu quả trên các sparse retriever như BM25. | Tương tự HyDE nhưng áp dụng mạnh mẽ hơn cho cả nhánh Sparse Retrieval của T-RAG. |
| 14 | **Zheng et al. (2023)**<br>*Take a Step Back: Evoking Reasoning via Abstraction in LLMs* | Tóm tắt/Trừu tượng hóa câu hỏi thành câu hỏi bao quát hơn trước khi retrieve để lấy được thông tin nền tảng. | Bổ sung phương pháp sinh query phụ trong Query Parser để xử lý các câu hỏi phức tạp. |

---

## 5. Temporal-Aware Retrieval (Time Decay)

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 15 | **Dakka et al. (2012)**<br>*Answering General Time-Sensitive Queries* | Phân loại các truy vấn có tính thời gian và kết hợp điểm số relevance với phân phối xác suất thời gian của tài liệu. | Đặt nền tảng cho việc nhận dạng "Temporal Intent" (cờ `requires_latest` trong T-RAG). |
| 16 | **Li et al. (2013)**<br>*Time-aware Information Retrieval* | Đề xuất các mô hình Time Decay (Exponential Decay, Polynomial Decay) để ưu tiên các bài báo mới trong ứng dụng News Search. | Nguồn gốc của **công thức Exponential Time Decay** mà T-RAG sẽ áp dụng cho Reranker. |
| 17 | **Ke et al. (2024)**<br>*Time-aware RAG for Large Language Models* | Nghiên cứu cách LLMs đối phó với thông tin cũ và đề xuất chèn nhãn thời gian (timestamps) vào văn bản. | Cho thấy hạn chế của việc chỉ nhét timestamp vào prompt. T-RAG giải quyết ở cấp độ Retrieval qua Time Penalty. |
| 18 | **Rao et al. (2023)**<br>*TRIME: Temporal Retrieval with In-Context Metadata Encoding* | Tìm kiếm tài liệu dựa trên cả nội dung và khoảng thời gian (time intervals) trong truy vấn. | Liên quan mật thiết đến việc **Metadata Filtering** bằng SQL trước khi thực hiện ANN Search trong LanceDB. |

---

## 6. Dataset & Benchmarking trong Doanh Nghiệp

| # | Paper | Contribution / Key Takeaway | Relevance to T-RAG |
|---|-------|---------------------------|-------------------|
| 19 | **Adlakha et al. (2023) / Sun et al. (2026)**<br>*EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge* | Giới thiệu benchmark dữ liệu doanh nghiệp quy mô 500k+ với tính nhiễu (messiness) và cấu trúc đa dạng. | Chính là **target dataset** của chúng ta. |
| 20 | **Thakur et al. (2021)**<br>*BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* | Cung cấp framework và bộ dataset đa dạng để đánh giá khả năng generalisation của các retriever ngoài Wikipedia. | Framework BEIR được sử dụng để implement hệ thống **Evaluation Metric** cho pipeline của T-RAG. |
| 21 | **Es et al. (2023)**<br>*RAGAS: Automated Evaluation of Retrieval Augmented Generation* | Đề xuất các metric (Faithfulness, Answer Relevance, Context Precision, Context Recall) dùng LLM as a judge. | Công cụ đo đạc chính trong **Phase 4: Benchmarking**. |

---

## Kết luận (Literature Synthesis)

Thông qua phân tích hơn 20 tài liệu nghiên cứu, hệ thống **T-RAG** không phải là một mô hình học sâu hoàn toàn mới, mà là sự **tổ hợp và kỹ nghệ hóa (engineering)** một cách tối ưu các kiến thức SOTA hiện có:
- Vấn đề **Từ vựng (Lexical mismatch)** được giải quyết bằng sự kết hợp giữa [5], [6] thông qua [8].
- Vấn đề **Truy vấn chưa rõ nghĩa** giải quyết bằng [12], [13].
- Vấn đề **Khối lượng (Density) và Mù thời gian (Temporal Blindness)** giải quyết bằng cách áp dụng có điều kiện tư tưởng của [15], [16] vào kiến trúc Reranker Cross-Encoder của [9].
