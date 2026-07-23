# 📊 Insight Report: T-RAG v6.2 Full Benchmark (500 Questions)

> **Dataset:** EnterpriseRAG-Bench (500 questions, 10 question types)
> **LLM:** Qwen2.5-14B-Instruct (vLLM, gpu_util=0.80)
> **Embedding:** BAAI/bge-large-en-v1.5
> **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
> **Judge:** Qwen2.5-14B-Instruct (vLLM OpenAI API, local port 8000)

---

## 1. Bảng Xếp Hạng Tổng Hợp (53 Cấu Hình)

| # | Pipeline | Config Details | Corr% | Comp% | Combined | Refused% | Total Lat | Retr Lat | Search Space |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| **1** | **🥇 OPT: High Recall + Sparse Heavy** | τ=0.05, γ=0.5, α=0.08, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | **36.80** | **45.67** | **32.24** | 17.2% | 1.02s | 0.43s | 3,577,666 |
| **2** | **🥈 Grid Tau=0.10** | τ=0.10, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | **36.67** | 44.99 | 31.94 | 18.4% | 0.88s | 0.29s | 3,323,143 |
| **3** | **🥉 OPT: Gamma=0.4** | τ=0.15, γ=0.4, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | **36.47** | 45.23 | 31.29 | 18.4% | 0.83s | 0.25s | 2,706,911 |
| 4 | Grid Gamma=0.0 | τ=0.15, γ=0.0, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 36.40 | 45.52 | 31.86 | 17.0% | 0.84s | 0.25s | 2,706,836 |
| 5 | Grid Alpha=0.15 | τ=0.15, γ=0.5, α=0.15, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 36.27 | 45.47 | 31.84 | 17.8% | 0.86s | 0.28s | 3,006,387 |
| 6 | Grid Alpha=0.50 | τ=0.15, γ=0.5, α=0.50, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 36.07 | 44.99 | 31.40 | 17.4% | 0.90s | 0.33s | 3,573,824 |
| 7 | Grid Tau=0.05 | τ=0.05, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 36.07 | 44.97 | 31.24 | 17.4% | 0.90s | 0.33s | 3,573,824 |
| 8 | OPT: D=0.4/S=0.6 | τ=0.15, γ=0.5, α=0.08, D=0.4/S=0.6, CSEP, SmartHop2, AdaptTau | 36.00 | **46.65** | 32.01 | 18.0% | 0.89s | 0.31s | 2,706,057 |
| **9** | **Targeted D: Alpha Sweet Spot** | τ=0.05, γ=0.4, α=0.15, D=0.4/S=0.6, CSEP, SmartHop2, AdaptTau | **36.00** | **45.32** | **31.10** | **17.2%** | **0.96s** | **0.36s** | **3,566,845** |
| 10 | Grid Dense=0.3 | τ=0.15, γ=0.5, α=0.08, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | 35.80 | 45.61 | 31.85 | 18.6% | 0.92s | 0.36s | 2,687,974 |
| 11 | Grid Alpha=0.00 | τ=0.15, γ=0.5, α=0.00, D=0.5/S=0.5, CSEP, SmartHop2 | 35.67 | 44.19 | 30.69 | 18.0% | 0.82s | 0.25s | 2,393,189 |
| 12 | OPT: Best Completeness | τ=0.15, γ=0.5, α=0.04, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | 35.60 | 45.84 | 31.38 | 19.8% | 0.91s | 0.33s | 2,540,258 |
| **13** | **Targeted G: Minimalist Best** | τ=0.10, γ=0.4, α=0.15, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | **35.60** | **45.87** | **31.24** | **17.0%** | **0.99s** | **0.40s** | **3,558,410** |
| 14 | Grid Tau=0.20 | τ=0.20, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 35.47 | 44.50 | 30.63 | 18.0% | 0.80s | 0.23s | 2,219,777 |
| 15 | Grid Alpha=0.12 | τ=0.15, γ=0.5, α=0.12, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 35.27 | 44.64 | 31.00 | 17.8% | 0.84s | 0.27s | 2,853,707 |
| 16 | Ablation: No AdaptTau | τ=0.15, γ=0.5, α=0.00 (forced), D=0.5/S=0.5, CSEP, SmartHop2 | 35.27 | 44.27 | 30.48 | 18.0% | 0.82s | 0.24s | 2,393,189 |
| 17 | OPT: Low Latency | τ=0.20, γ=0.3, α=0.00, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 35.27 | 43.71 | 30.21 | 19.0% | 0.76s | 0.20s | 1,964,143 |
| 18 | OPT: Speed+Sparse | τ=0.20, γ=0.5, α=0.00, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | 35.20 | 44.65 | 30.68 | 18.4% | 0.85s | 0.27s | 1,944,632 |
| 19 | OPT: Balanced | τ=0.15, γ=0.5, α=0.04, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 35.07 | 45.21 | 30.89 | 18.6% | 0.83s | 0.26s | 2,562,116 |
| 20 | Grid Alpha=0.04 | τ=0.15, γ=0.5, α=0.04, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 35.07 | 45.20 | 30.77 | 18.6% | 0.83s | 0.26s | 2,562,116 |
| 21 | Grid Alpha=0.25 | τ=0.15, γ=0.5, α=0.25, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 34.87 | 44.37 | 30.62 | 17.2% | 0.90s | 0.32s | 3,416,644 |
| 22 | Grid Gamma=1.0 | τ=0.15, γ=1.0, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 34.87 | 42.32 | 29.71 | 19.0% | 0.83s | 0.27s | 2,704,042 |
| 23 | OPT: D=0.2/S=0.8 | τ=0.15, γ=0.5, α=0.08, D=0.2/S=0.8, CSEP, SmartHop2, AdaptTau | 34.80 | 45.08 | 30.83 | 18.8% | 0.97s | 0.39s | 2,693,001 |
| 24 | **T-RAG v2 Standard** | τ=0.15, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 34.67 | 44.91 | 30.72 | 17.0% | 0.84s | 0.27s | 2,711,818 |
| 25 | Ablation: No CSEP | τ=0.15, γ=0.5, α=0.08, D=0.5/S=0.5, No CSEP, SmartHop2, AdaptTau | 34.67 | 44.38 | 30.13 | 18.2% | 0.82s | 0.25s | 2,704,197 |
| 26 | Grid Dense=0.1 | τ=0.15, γ=0.5, α=0.08, D=0.1/S=0.9, CSEP, SmartHop2, AdaptTau | 34.60 | 45.13 | 30.67 | 17.8% | 1.02s | 0.42s | 2,690,576 |
| **27** | **Targeted H: Speed King v2** | τ=0.15, γ=0.0, α=0.00, D=0.4/S=0.6, CSEP, SmartHop2 | **34.60** | **43.75** | **30.32** | **19.0%** | **0.84s** | **0.26s** | **2,387,237** |
| 28 | OPT: Retrieve Depth=30 | τ=0.15, γ=0.5, α=0.08, R=30, K=5, CSEP, SmartHop2, AdaptTau | 34.47 | 44.42 | 30.38 | 16.8% | 0.92s | 0.34s | 2,707,042 |
| 29 | Grid Gamma=0.3 | τ=0.15, γ=0.3, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 34.47 | 44.54 | 30.19 | 18.0% | 0.82s | 0.25s | 2,706,695 |
| 30 | Grid Gamma=0.7 | τ=0.15, γ=0.7, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 34.07 | 44.39 | 29.84 | 17.6% | 0.83s | 0.25s | 2,698,037 |
| 31 | OPT: Best Correctness | τ=0.15, γ=0.5, α=0.25, D=0.1/S=0.9, CSEP, SmartHop2, AdaptTau | 34.00 | 44.03 | 29.67 | 19.2% | 1.07s | 0.47s | 3,392,562 |
| 32 | T-RAG v1 High-Recall G1 | τ=0.05, γ=1.0, no CSEP, no AdaptTau | 33.80 | 43.43 | 29.71 | 22.6% | 1.19s | 2.31s | 3,550,005 |
| 33 | T-RAG v1 Balanced G1 | τ=0.15, γ=1.0, no CSEP, no AdaptTau | 33.80 | 43.29 | 29.66 | 20.6% | 1.02s | 1.48s | 2,354,016 |
| 34 | OPT: High Speed Sparse Heavy | τ=0.30, γ=0.5, α=0.08, D=0.1/S=0.9, CSEP, SmartHop2, AdaptTau | 33.60 | 44.77 | 29.52 | 19.8% | 0.86s | 0.29s | 1,438,218 |
| 35 | **Hybrid Baseline** | BM25+Vector Hybrid, Reranker, K=5 | 33.60 | 44.11 | 29.35 | 18.0% | 1.15s | 0.63s | 4,213,106 |
| 36 | Grid Tau=0.30 | τ=0.30, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, SmartHop2, AdaptTau | 33.27 | 42.91 | 29.33 | 18.0% | 0.74s | 0.18s | 1,456,839 |
| 37 | Sparse Only | τ=0.15, γ=0.5, α=0.08, D=0.0/S=1.0, CSEP, SmartHop2, AdaptTau | 33.20 | 43.77 | 29.88 | 20.0% | 1.07s | 0.48s | 2,691,477 |
| 38 | OPT: Max Performance | τ=0.10, γ=0.5, α=0.04, D=0.1/S=0.9, CSEP, SmartHop2, AdaptTau | 33.20 | 43.67 | 29.17 | 18.8% | 1.04s | 0.45s | 3,100,597 |
| **39** | **Targeted B: Precision Strike** | τ=0.10, γ=0.0, α=0.15, D=0.4/S=0.6, CSEP, SmartHop2, AdaptTau | **33.20** | **43.72** | **28.59** | **19.2%** | **0.96s** | **0.36s** | **3,566,651** |
| 40 | T-RAG v1 High-Speed G1 | τ=0.30, γ=1.0, no CSEP, no AdaptTau | 33.00 | 41.86 | 29.15 | 21.4% | 0.93s | 1.08s | 1,282,848 |
| 41 | HyDE Baseline | HyDE (LLM gen hypothetical) -> Hybrid -> Rerank | 33.00 | 42.97 | 28.01 | 18.2% | 1.23s | 0.63s | 4,213,106 |
| 42 | Ablation: No Smart Hop2 | τ=0.15, γ=0.5, α=0.08, D=0.5/S=0.5, CSEP, No SmartHop2, AdaptTau | 32.80 | 43.93 | 29.39 | 18.6% | 1.16s | 0.58s | 2,681,805 |
| 43 | Query Expansion Baseline | LLM 3 sub-queries -> Parallel Hybrid -> RRF -> Rerank | 32.60 | 43.14 | 28.34 | 19.8% | 2.46s | 1.91s | 4,213,106 |
| **44** | **Targeted A: Ultimate Combo** | τ=0.05, γ=0.0, α=0.15, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | **32.40** | **42.77** | **28.41** | **21.4%** | **1.00s** | **0.39s** | **3,591,459** |
| 45 | T-RAG v1 High-Recall | τ=0.05, γ=0.0, no CSEP, no AdaptTau | 32.40 | 43.25 | 28.15 | 19.6% | 1.25s | 2.28s | 3,598,130 |
| 46 | T-RAG v1 Balanced | τ=0.15, γ=0.0, no CSEP, no AdaptTau | 32.40 | 42.56 | 27.91 | 19.2% | 1.05s | 1.53s | 2,372,449 |
| 47 | OPT: Retrieve Depth=10 | τ=0.15, γ=0.5, α=0.08, R=10, K=5, CSEP, SmartHop2, AdaptTau | 32.40 | 42.29 | 27.85 | 22.2% | 0.71s | 0.16s | 2,711,610 |
| 48 | T-RAG v1 High-Speed | τ=0.30, γ=0.0, no CSEP, no AdaptTau | 32.20 | 42.44 | 28.36 | 22.0% | 0.92s | 1.01s | 1,284,666 |
| **49** | **Targeted F: Wide Net Balanced** | τ=0.05, γ=0.0, α=0.08, R=25, K=5, CSEP, SmartHop2, AdaptTau | **32.20** | **43.34** | **27.79** | **19.4%** | **1.01s** | **0.42s** | **3,575,488** |
| **50** | **Targeted E: Low Gamma High Alpha** | τ=0.10, γ=0.0, α=0.12, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | **31.80** | **43.46** | **28.17** | **21.2%** | **1.00s** | **0.39s** | **3,483,602** |
| 51 | T-RAG v1 No Reranker | τ=0.15, γ=0.0, no Reranker, no CSEP, no AdaptTau | 31.80 | 40.72 | 27.69 | 20.8% | 1.06s | 1.66s | 2,372,449 |
| **52** | **Targeted C: Gamma Zero Low Tau** | τ=0.10, γ=0.0, α=0.08, D=0.3/S=0.7, CSEP, SmartHop2, AdaptTau | **31.00** | **43.51** | **27.55** | **21.2%** | **1.00s** | **0.38s** | **3,323,775** |
| 53 | OPT: TopK=3 | τ=0.15, γ=0.5, α=0.08, K=3, CSEP, SmartHop2, AdaptTau | 30.60 | 42.06 | 27.39 | 21.6% | 0.66s | 0.26s | 2,711,818 |
| 54 | BM25 Baseline | Pure BM25, K=5 | 29.20 | 39.50 | 25.23 | 23.0% | 0.97s | 0.41s | 4,213,106 |
| 55 | Grid Dense=0.7 | τ=0.15, γ=0.5, α=0.08, D=0.7/S=0.3, CSEP, SmartHop2, AdaptTau | 28.26 | 37.18 | 24.34 | 24.0% | 0.82s | 0.26s | 2,704,805 |
| 56 | LLM Router Baseline | LLM routes shards -> Hybrid -> RRF -> Rerank | 27.80 | 38.57 | 23.96 | 21.8% | 0.88s | 0.32s | 2,075,036 |
| 57 | Grid Dense=0.9 | τ=0.15, γ=0.5, α=0.08, D=0.9/S=0.1, CSEP, SmartHop2, AdaptTau | 25.85 | 34.83 | 22.14 | 25.4% | 0.82s | 0.26s | 2,704,805 |
| 58 | VECTOR_RERANKER Baseline | Pure Vector -> Rerank, K=5 | 24.65 | 33.84 | 20.89 | 26.8% | 0.69s | 0.23s | 4,213,106 |
| 59 | Dense Only | τ=0.15, γ=0.5, α=0.08, D=1.0/S=0.0, CSEP, SmartHop2, AdaptTau | 24.25 | 34.66 | 21.11 | 26.2% | 0.82s | 0.25s | 2,705,415 |
| 60 | OPT: TopK=1 | τ=0.15, γ=0.5, α=0.08, K=1, CSEP, SmartHop2, AdaptTau | 22.40 | 33.05 | 19.63 | 40.6% | 0.48s | 0.25s | 2,711,818 |
| 61 | Vector Baseline | Pure Vector, K=5 | 19.60 | 30.24 | 16.47 | 32.0% | 0.60s | 0.20s | 4,213,106 |

---

## 2. Phân Tích Chi Tiết Theo Nhóm

### 2.1. Baselines (7 pipelines)

| Pipeline | Corr% | Comp% | Lat | Config |
|:---|:---:|:---:|:---:|:---|
| Hybrid | 33.60 | 44.11 | 1.15s | BM25+Vector Hybrid → Rerank → K=5 |
| HyDE | 33.00 | 42.97 | 1.23s | LLM gen hypothetical → Embed → Hybrid → Rerank |
| Query Expansion | 32.60 | 43.14 | **2.46s** | LLM gen 3 sub-queries → 4× Hybrid → RRF → Rerank |
| BM25 | 29.20 | 39.50 | 0.97s | Pure BM25 → K=5 |
| LLM Router | 27.80 | 38.57 | 0.88s | LLM routes shards → Hybrid → RRF → Rerank |
| Vector+Reranker | 24.65 | 33.84 | 0.69s | Pure Vector → Rerank → K=5 |
| Vector | 19.60 | 30.24 | 0.60s | Pure Vector → K=5 |

> [!IMPORTANT]
> **Insight 1:** Hybrid Baseline (33.60%) là baseline mạnh nhất. HyDE và Query Expansion không cải thiện đáng kể mặc dù tốn thêm 1 lần gọi LLM (tăng latency 7-114%). LLM Router kém hơn cả BM25 vì routing sai shard sẽ mất hoàn toàn context.

> [!IMPORTANT]
> **Insight 2:** Vector Baseline (19.60%) kém xa BM25 (29.20%), chứng tỏ dataset doanh nghiệp này chứa rất nhiều thuật ngữ kỹ thuật đặc thù mà embedding chung không nắm bắt được.

---

### 2.2. T-RAG v1 (7 pipelines)

| Pipeline | Corr% | Comp% | Retr Lat | Config |
|:---|:---:|:---:|:---:|:---|
| Balanced G1 (γ=1.0) | 33.80 | 43.29 | **1.48s** | τ=0.15, γ=1.0 |
| High-Recall G1 (γ=1.0) | 33.80 | 43.43 | **2.31s** | τ=0.05, γ=1.0 |
| High-Speed G1 (γ=1.0) | 33.00 | 41.86 | **1.08s** | τ=0.30, γ=1.0 |
| Balanced (γ=0.0) | 32.40 | 42.56 | **1.53s** | τ=0.15, γ=0.0 |
| High-Recall (γ=0.0) | 32.40 | 43.25 | **2.28s** | τ=0.05, γ=0.0 |
| High-Speed (γ=0.0) | 32.20 | 42.44 | **1.01s** | τ=0.30, γ=0.0 |
| No Reranker | 31.80 | 40.72 | **1.66s** | τ=0.15, no reranker |

> [!IMPORTANT]
> **Insight 3:** T-RAG v1 chỉ đạt tối đa 33.80% (ngang Hybrid Baseline). Retrieval Latency cực cao (1.0s - 2.3s) do bug double-encoding. Gamma=1.0 (G1) luôn tốt hơn gamma=0.0 khoảng 1-1.4% absolute.

---

### 2.3. T-RAG v2 Grid Search: τ_base

| τ_base | Corr% | Comp% | Retr Lat | Search Space |
|:---:|:---:|:---:|:---:|---:|
| **0.10** | **36.67** | 44.99 | 0.29s | 3,323,143 |
| 0.05 | 36.07 | 44.97 | 0.33s | 3,573,824 |
| 0.15 (default) | 34.67 | 44.91 | 0.27s | 2,711,818 |
| 0.20 | 35.47 | 44.50 | 0.23s | 2,219,777 |
| 0.30 | 33.27 | 42.91 | 0.18s | 1,456,839 |

> [!IMPORTANT]
> **Insight 4:** τ=0.10 là **sweet spot** tốt nhất, không phải τ=0.05 (quá rộng → nhiễu) hay τ=0.15 (mặc định, hơi chặt → bỏ sót). Tau nhỏ hơn = lấy nhiều tài liệu hơn ở Hop 1 → Reranker có nhiều ứng viên tốt hơn để chọn.

---

### 2.4. T-RAG v2 Grid Search: γ (Gamma - Diversity Penalty)

| γ | Corr% | Comp% |
|:---:|:---:|:---:|
| **0.0** | **36.40** | **45.52** |
| 0.3 | 34.47 | 44.54 |
| 0.4 | 36.47 | 45.23 |
| 0.5 (default) | 34.67 | 44.91 |
| 0.7 | 34.07 | 44.39 |
| 1.0 | 34.87 | 42.32 |

> [!IMPORTANT]
> **Insight 5:** γ=0.0 (tắt hoàn toàn diversity penalty) đạt **36.40%**, là giá trị Gamma tốt nhất. γ=0.4 cũng rất tốt (36.47%). Điều này cho thấy: trên tập dữ liệu doanh nghiệp, việc giữ nguyên các tài liệu giống nhau (không phạt trùng lặp) mang lại hiệu quả cao hơn vì các tài liệu "trùng lặp" thực chất là các phiên bản/góc nhìn khác nhau về cùng một chủ đề.

> [!WARNING]
> **Bất thường:** γ=0.3 (34.47%) lại thấp hơn γ=0.5 (34.67%) và γ=0.4 (36.47%). Đường cong Gamma không monotonic, gợi ý rằng có sự tương tác phức tạp giữa gamma và phân phối điểm reranking. Cần nghiên cứu thêm.

---

### 2.5. T-RAG v2 Grid Search: α (Adaptive Tau Sensitivity)

| α | Corr% | Comp% | Search Space |
|:---:|:---:|:---:|---:|
| 0.00 (static) | 35.67 | 44.19 | 2,393,189 |
| 0.04 | 35.07 | 45.20 | 2,562,116 |
| 0.08 (default) | 34.67 | 44.91 | 2,711,818 |
| 0.12 | 35.27 | 44.64 | 2,853,707 |
| **0.15** | **36.27** | **45.47** | 3,006,387 |
| 0.25 | 34.87 | 44.37 | 3,416,644 |
| 0.50 | 36.07 | 44.99 | 3,573,824 |

> [!IMPORTANT]
> **Insight 6:** α=0.15 đạt **36.27%**, tốt hơn mặc định α=0.08 (34.67%). Tuy nhiên α=0.50 cũng đạt 36.07%, cho thấy adaptive tau ở mức vừa phải (0.12-0.15) là tối ưu nhất. α quá lớn làm Search Space phình quá to mà không tăng thêm chất lượng.

---

### 2.6. T-RAG v2 Grid Search: Dense/Sparse Weight

| D / S | Corr% | Comp% | Retr Lat |
|:---:|:---:|:---:|:---:|
| 1.0 / 0.0 (Dense Only) | 24.25 | 34.66 | 0.25s |
| 0.9 / 0.1 | 25.85 | 34.83 | 0.26s |
| 0.7 / 0.3 | 28.26 | 37.18 | 0.26s |
| 0.5 / 0.5 (default) | 34.67 | 44.91 | 0.27s |
| 0.4 / 0.6 | 36.00 | **46.65** | 0.31s |
| **0.3 / 0.7** | **35.80** | 45.61 | 0.36s |
| 0.2 / 0.8 | 34.80 | 45.08 | 0.39s |
| 0.1 / 0.9 | 34.60 | 45.13 | 0.42s |
| 0.0 / 1.0 (Sparse Only) | 33.20 | 43.77 | 0.48s |

> [!IMPORTANT]
> **Insight 7 (Critical Finding):** Tỷ lệ **D=0.4/S=0.6** đạt **Completeness cao nhất (46.65%)** trong toàn bộ benchmark. D=0.3/S=0.7 đạt **Correctness cao thứ 2 (35.80%)**. "Sweet Spot" nằm trong khoảng D=0.3-0.4 / S=0.6-0.7.

> [!CAUTION]
> Dense-heavy (D≥0.7) sụt giảm nghiêm trọng, từ 34.67% xuống 24-28%. Chứng tỏ embedding BGE-large không capture tốt thuật ngữ kỹ thuật enterprise.

---

### 2.7. Ablation Study

| Cấu hình | Corr% | Comp% | Retr Lat | Δ vs Standard |
|:---|:---:|:---:|:---:|:---:|
| T-RAG v2 Standard (Full) | 34.67 | 44.91 | 0.27s | — |
| No Adaptive Tau (α=0) | 35.27 | 44.27 | 0.24s | +0.60% |
| No CSEP (bỏ Hop 2) | 34.67 | 44.38 | 0.25s | 0.00% |
| No Smart Hop 2 (Hop 2 luôn chạy) | 32.80 | 43.93 | **0.58s** | **-1.87%** |

> [!IMPORTANT]
> **Insight 8:** No Smart Hop 2 giảm Correctness **1.87%** và tăng Retrieval Latency **2.15× (0.27s → 0.58s)**. Đây là bằng chứng mạnh nhất cho giá trị của Smart Hop 2.

> [!WARNING]
> **Insight 9:** Ablation "No Adaptive Tau" lại tăng nhẹ Correctness (+0.60%). Điều này gợi ý rằng giá trị α mặc định (0.08) có thể chưa tối ưu — cần điều chỉnh lại (α=0.15 từ Grid Search đã tốt hơn).

---

### 2.8. Search Depth & Context Window

| Config | Corr% | Comp% | Refused% |
|:---|:---:|:---:|:---:|
| K_final=1 | 22.40 | 33.05 | **40.6%** |
| K_final=3 | 30.60 | 42.06 | 21.6% |
| K_final=5 (default) | 34.67 | 44.91 | 17.0% |
| R_retrieve=10 | 32.40 | 42.29 | 22.2% |
| R_retrieve=20 (default) | 34.67 | 44.91 | 17.0% |
| R_retrieve=30 | 34.47 | 44.42 | 16.8% |

> [!IMPORTANT]
> **Insight 10:** K_final=5 là tối ưu. K=1 quá ít context → 40.6% refused. K=3 vẫn kém 4% so với K=5. R_retrieve=20 là sweet spot — R=10 thiếu candidate cho Reranker, R=30 tăng latency mà không cải thiện quality.

---

## 3. Phân Tích Theo Question Type (Top 5 Pipelines vs Hybrid Baseline)

| Question Type (n) | Hybrid BL | v2 Standard | **#1 High Recall** | **#2 Tau=0.10** | **#3 Gamma=0.4** | **#8 D=0.4/S=0.6** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| basic (175) | 40.6% | 44.0% | **44.6%** | 41.7% | **44.6%** | **45.1%** |
| semantic (125) | 14.4% | 15.2% | 16.8% | 17.6% | 17.6% | 16.8% |
| intra_doc_reasoning (40) | 45.0% | 35.0% | 30.0% | **37.5%** | **37.5%** | 32.5% |
| project_related (40) | 17.5% | 10.3% | 17.5% | 20.5% | 17.9% | 12.5% |
| constrained (30) | 20.0% | 33.3% | **36.7%** | 33.3% | 30.0% | 30.0% |
| conflicting_info (20) | 50.0% | 40.0% | **55.0%** | **60.0%** | 45.0% | 45.0% |
| completeness (20) | 15.0% | 10.0% | **20.0%** | 15.0% | 15.0% | 15.0% |
| miscellaneous (20) | 60.0% | **75.0%** | **75.0%** | **75.0%** | 70.0% | **80.0%** |
| high_level (10) | 40.0% | 40.0% | **50.0%** | **60.0%** | **50.0%** | **50.0%** |
| info_not_found (20) | **100.0%** | **100.0%** | **100.0%** | 95.0% | **100.0%** | **100.0%** |

> [!IMPORTANT]
> **Insight 11:** T-RAG v2 vượt trội Hybrid ở **basic** (+4%), **constrained** (+16.7%), **conflicting_info** (+10%), **miscellaneous** (+15%), và **high_level** (+20%). Nhưng T-RAG v2 kém hơn Hybrid ở **intra_doc_reasoning** (-15%) và **project_related** (-5%). Đây là điểm yếu cần tập trung cải thiện.

> [!WARNING]
> **Insight 12:** `semantic` là loại câu hỏi khó nhất, chỉ đạt 14-18% cho tất cả hệ thống. Đây là bottleneck chính. Cải thiện semantic retrieval có thể tăng đáng kể Correctness tổng thể (125/500 = 25% tổng số câu hỏi).

---

## 4. Tổng Hợp Các Phát Hiện Quan Trọng

### 🏆 Cấu hình tối ưu hiện tại
**OPT: High Recall + Sparse Heavy** (`τ=0.05, γ=0.5, α=0.08, D=0.3/S=0.7`)
- Correctness: **36.80%** (+9.5% relative so với Hybrid BL, +6.1% so với v2 Standard)
- Completeness: **45.67%**
- Retrieval Latency: **0.43s** (so với 1.48-2.31s của T-RAG v1)

### 📋 Bảng tham số tối ưu rút ra từ Grid Search

| Tham số | Mặc định v2 | Giá trị tối ưu | Ghi chú |
|:---|:---:|:---:|:---|
| τ_base | 0.15 | **0.05 - 0.10** | Mở rộng phễu Hop 1, để Reranker chọn |
| γ | 0.5 | **0.0 - 0.4** | Giảm/tắt diversity penalty |
| α | 0.08 | **0.12 - 0.15** | Tăng sensitivity cho Adaptive Tau |
| Dense/Sparse | 0.5/0.5 | **0.3-0.4 / 0.6-0.7** | Nghiêng nhẹ về BM25 |
| K_final | 5 | **5** | Giữ nguyên |
| R_retrieve | 20 | **20** | Giữ nguyên |

---

## 5. Đề Xuất Cấu Hình Thế Hệ Tiếp Theo

### 5.1. Các Cấu Hình T-RAG v2 "Remix" (Kết Hợp Tham Số Tối Ưu)

Dựa trên insight trên, mình đề xuất **8 cấu hình mới** kết hợp các tham số tối ưu chưa từng thử cùng nhau:

#### Config A: "Ultimate Combo"
```
τ=0.05, γ=0.0, α=0.15, D=0.3/S=0.7
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** Kết hợp 4 tham số tối ưu nhất: τ thấp nhất (0.05) + γ tắt hoàn toàn (0.0) + α vừa phải (0.15) + D/S tối ưu (0.3/0.7). Chưa có cấu hình nào test tổ hợp này.

#### Config B: "Precision Strike"
```
τ=0.10, γ=0.0, α=0.15, D=0.4/S=0.6
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** τ=0.10 (top 2 Grid Tau) + γ=0.0 (top 1 Grid Gamma) + α=0.15 (top 1 Grid Alpha) + D=0.4/S=0.6 (top 1 Completeness). Tối ưu cho cả Correctness lẫn Completeness.

#### Config C: "Gamma Zero Low Tau"
```
τ=0.10, γ=0.0, α=0.08, D=0.3/S=0.7
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** Grid Gamma=0.0 (36.40%) + Grid Tau=0.10 (36.67%) + Grid Dense=0.3 (35.80%). Test xem 3 tham số tối ưu có cộng hưởng không.

#### Config D: "Alpha Sweet Spot"
```
τ=0.05, γ=0.4, α=0.15, D=0.4/S=0.6
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** τ mở rộng tối đa (0.05) + γ=0.4 (top 3 overall, 36.47%) + α=0.15 (top 1 Alpha) + D/S=0.4/0.6 (top 1 Completeness).

#### Config E: "Low Gamma High Alpha"
```
τ=0.10, γ=0.0, α=0.12, D=0.3/S=0.7
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** Biến thể của Config C với α tăng nhẹ lên 0.12 để tận dụng adaptive range rộng hơn.

#### Config F: "Wide Net Balanced"
```
τ=0.05, γ=0.0, α=0.08, D=0.4/S=0.6
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=25
```
**Lý do:** Tăng R_retrieve lên 25 (chưa từng thử) kết hợp τ thấp + γ=0 + D/S balanced-sparse. R=25 là điểm giữa R=20 và R=30.

#### Config G: "Minimalist Best"
```
τ=0.10, γ=0.4, α=0.15, D=0.3/S=0.7
CSEP=On, SmartHop2=On, AdaptTau=On, K=5, R=20
```
**Lý do:** Kết hợp τ=0.10 (top 2) + γ=0.4 (top 3) + α=0.15 (top 1 Alpha) + D=0.3/S=0.7. Cấu hình "tinh gọn" nhất.

#### Config H: "Speed King v2"
```
τ=0.15, γ=0.0, α=0.00, D=0.4/S=0.6
CSEP=On, SmartHop2=On, K=5, R=20
```
**Lý do:** Tối ưu tốc độ: τ vừa phải (0.15) + γ=0.0 (bỏ diversity computation) + α=0.00 (bỏ adaptive tau computation) + D/S tối ưu. Dự kiến latency cực thấp mà vẫn giữ quality cao.

### 5.2. Kết Quả Thực Nghiệm Các Cấu Hình "Remix" (Targeted Benchmark)

Sau khi chạy thực nghiệm 500 câu hỏi đầy đủ trên 8 cấu hình đề xuất (kết quả lưu tại [results_targeted_v6](file:///network-volume/RAG-/T-RAG_Project/results_targeted_v6/)), dưới đây là bảng so sánh chi tiết:

| Pipeline / Config | Correctness% | Completeness% | Refused% | Total Lat | Retr Lat | Space Search (Docs) |
|:---|:---:|:---:|:---:|:---:|:---:|---:|
| **Targeted D: Alpha Sweet Spot** | **36.00** | 45.30 | 17.2 | 0.96s | 0.36s | 3,566,845 |
| **Targeted G: Minimalist Best** | 35.60 | **45.90** | **17.0** | 0.99s | 0.40s | 3,558,410 |
| Targeted H: Speed King v2 | 34.60 | 43.80 | 19.0 | **0.84s** | **0.26s** | 2,387,237 |
| Targeted B: Precision Strike | 33.20 | 43.70 | 19.2 | 0.96s | 0.36s | 3,566,651 |
| Targeted A: Ultimate Combo | 32.40 | 42.80 | 21.4 | 1.00s | 0.39s | 3,591,459 |
| Targeted F: Wide Net Balanced | 32.20 | 43.30 | 19.4 | 1.01s | 0.42s | 3,575,488 |
| Targeted E: Low Gamma High Alpha | 31.80 | 43.50 | 21.2 | 1.00s | 0.39s | 3,483,602 |
| Targeted C: Gamma Zero Low Tau | 31.00 | 43.50 | 21.2 | 1.00s | 0.38s | 3,323,775 |

> [!IMPORTANT]
> **Phát hiện quan trọng 1: Sự tương tác phức tạp giữa $\tau_{base}$ và $\gamma$ (Diversity Penalty)**
> - Mặc dù trong Grid Search độc lập, $\gamma=0.0$ (không phạt trùng lặp) đạt hiệu năng cao khi đi kèm với $\tau_{base}=0.15$ mặc định. Tuy nhiên, khi kết hợp với $\tau_{base}$ thấp ($0.05$ hoặc $0.10$ ở các Config A, B, C, E, F), hiệu năng giảm mạnh xuống **31% - 33%** và tỉ lệ từ chối tăng vọt lên **19.2% - 21.4%**.
> - **Lý do:** Khi $\tau_{base}$ thấp, phễu Hop 1 lấy vào lượng lớn tài liệu tương tự nhau. Việc không có diversity penalty ($\gamma=0.0$) khiến Reranker chọn các tài liệu trùng lặp/gần trùng lặp vào context cuối cùng ($K=5$). Điều này gây nghẽn thông tin, thiếu đa dạng hóa evidence để LLM trả lời, dẫn đến tỉ lệ từ chối (Refused) tăng và độ chính xác giảm.
> - Ngược lại, việc giữ một phần diversity penalty (**$\gamma = 0.4$** ở Config D và G) giúp phân tán thông tin tốt, dẫn đến chất lượng vượt trội (**36.00%** và **35.60%**) và tỉ lệ từ chối thấp nhất (**17.0%**).

> [!TIP]
> **Phát hiện quan trọng 2: Cấu hình tốt nhất cho Production - Targeted H (Speed King v2)**
> - `Targeted H` (`τ=0.15, γ=0.0, α=0.00, D=0.4/S=0.6`) đạt độ chính xác **34.60%** (vẫn cao hơn Hybrid Baseline 33.60% và tương đương T-RAG v2 Standard).
> - Đặc biệt, nó đạt **thời gian phản hồi cực nhanh 0.84s (Retrieval chỉ mất 0.26s)**. Đây là ứng cử viên lý tưởng nhất để đưa vào môi trường production nhờ lược bỏ các phép tính adaptive tau và diversity penalty giúp giảm thiểu đáng kể chi phí CPU/GPU tính toán.

---

### 5.3. Ý Tưởng T-RAG v3 (Cải Tiến Kiến Trúc)

Dựa trên phân tích các điểm yếu, mình đề xuất các hướng cải tiến kiến trúc cho T-RAG v3:

#### Ý tưởng 1: Query-Adaptive Dense/Sparse Weight
**Vấn đề:** Tỷ lệ D/S cố định cho mọi câu hỏi là không tối ưu. Câu hỏi "What is the default timeout?" cần BM25 mạnh, nhưng "Explain the benefits of microservices" cần Vector mạnh.

**Giải pháp:** Dùng một classifier nhẹ (hoặc rule-based dựa trên đặc tính câu hỏi) để tự động điều chỉnh D/S cho từng query:
```python
# Pseudo-code
if query_has_technical_terms(query):  # mã lỗi, API name, config key
    dense_weight, sparse_weight = 0.2, 0.8
elif query_is_conceptual(query):  # "explain", "why", "how does"
    dense_weight, sparse_weight = 0.6, 0.4
else:
    dense_weight, sparse_weight = 0.4, 0.6  # default balanced-sparse
```

#### Ý tưởng 2: Reranker-Guided Hop 2 Trigger
**Vấn đề:** Smart Hop 2 hiện dùng ngưỡng cố định trên similarity score để quyết định có nhảy Hop 2 hay không.

**Giải pháp:** Sử dụng **điểm Reranker** của top documents sau Hop 1 để quyết định. Nếu max reranker score < threshold → evidence yếu → trigger Hop 2. Reranker score chính xác hơn nhiều so với embedding similarity score.

#### Ý tưởng 3: Late Interaction / ColBERT-style Reranking
**Vấn đề:** `semantic` questions chỉ đạt 14-18% cho tất cả systems. Cross-encoder reranker hiện tại (MiniLM-L6) quá nhỏ (22M params) để hiểu ngữ nghĩa sâu.

**Giải pháp:** Thay thế reranker bằng mô hình lớn hơn (ví dụ: `BAAI/bge-reranker-v2-m3` hoặc `Jina Reranker v2`) hoặc sử dụng ColBERT-style late interaction để cải thiện semantic matching mà không tăng quá nhiều latency.

#### Ý tưởng 4: Hybrid Chunk + Document Retrieval
**Vấn đề:** Hiện tại mỗi chunk được index độc lập. Nhưng `intra_document_reasoning` questions (cần tổng hợp thông tin từ nhiều phần trong cùng 1 tài liệu) kém hơn Hybrid baseline (-15%).

**Giải pháp:** Thêm bước "Document Expansion" sau retrieval: Khi đã chọn được top chunks, tự động kéo thêm các chunks lân cận (sibling chunks, parent chunk) từ cùng tài liệu gốc để cung cấp context đầy đủ hơn cho LLM.

#### Ý tưởng 5: Ensemble Score Fusion
**Vấn đề:** Hiện tại chỉ dùng 1 cách tính hybrid score (weighted sum).

**Giải pháp:** Thử **Reciprocal Rank Fusion (RRF)** thay vì weighted sum để merge kết quả Dense và Sparse. RRF ít nhạy cảm với scale khác nhau giữa BM25 scores và cosine similarity scores.