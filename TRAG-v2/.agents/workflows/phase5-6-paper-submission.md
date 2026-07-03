# Workflow: Phase 5 & 6 — Paper Writing & Submission

## Trigger
Phase 4 done. Có đầy đủ số liệu, figures, và case studies.

## Goal
Bài báo hoàn chỉnh 8 trang (ACL format), đã qua review nội bộ, sẵn sàng submit.

---

## STEP 5.1 — Setup LaTeX Environment

### Tải ACL style

```bash
mkdir -p paper/
cd paper/

# Download ACL 2026 template
wget https://acl-org.github.io/ACLPUB/downloads/acl2026/acl-cite.bib
wget https://acl-org.github.io/ACLPUB/downloads/acl2026/acl.sty

# Hoặc dùng ECIR / SIGIR template tùy venue target
```

### Cấu trúc `paper/`

```
paper/
├── main.tex
├── references.bib
├── acl_latex.sty (hoặc venue style)
├── figures/
│   ├── trag_architecture.pdf      # Fig 1
│   ├── ablation_bar.pdf           # Fig 2
│   └── time_decay_illustration.pdf  # Fig 3 (optional)
├── tables/
│   ├── main_results.tex
│   └── ablation_results.tex
└── appendix.tex (optional)
```

---

## STEP 5.2 — Thứ tự viết (theo quy tắc enterprise research)

### Bước 1: Viết Section 5 — Experiments (TRƯỚC TIÊN)

```latex
\section{Experiments}

\subsection{Dataset}
We evaluate on EnterpriseRAG-Bench~\cite{...}, a large-scale enterprise
dataset containing over 500,000 documents from 9 enterprise sources
including Slack (285k), Gmail (121k), Linear/Jira (41k), and Confluence (5k).

\subsection{Baselines}
We compare T-RAG against the following baselines:
\begin{itemize}
    \item \textbf{Standard RAG}: Dense retrieval with top-50 ANN search...
    \item \textbf{BM25-only}: Sparse retrieval using Okapi BM25...
    \item \textbf{Dense-only}: Embedding-based retrieval without sparse...
\end{itemize}

\subsection{Metrics}
We report MRR@10, Recall@5, and NDCG@10 for retrieval quality,
and end-to-end latency (P50) for efficiency.

\subsection{Main Results}
Table~\ref{tab:main_results} shows the main results...
\input{tables/main_results}

\subsection{Ablation Study}
Table~\ref{tab:ablation} presents our ablation study...
\input{tables/ablation_results}

\subsection{Analysis}
\subsubsection{Temporal Query Performance}
T-RAG achieves X\% improvement on temporal queries...

\subsubsection{Case Study}
...
```

### Bước 2: Viết Section 4 — Methodology

```latex
\section{Methodology: T-RAG}

\subsection{Overview}
Figure~\ref{fig:architecture} illustrates the T-RAG pipeline...

\subsection{Unified Storage with LanceDB}
...

\subsection{Self-Query Expansion}
...

\subsection{Hybrid Retrieval}
We combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF):
\begin{equation}
    \text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
\end{equation}
where $k=60$ is the RRF constant and $R$ is the set of ranked lists.

\subsection{Conditional Temporal Reranking}
We propose a conditional time decay reranking function:
\begin{equation}
    \text{Score}(d) = \text{Relevance}(d) \times e^{-\lambda \Delta t}
\end{equation}
where $\lambda > 0$ only when \texttt{requires\_latest=True}...
```

### Bước 3: Viết Section 2 — Related Work

Dựa trên `docs/lit_review/literature_review.md`

```latex
\section{Related Work}

\paragraph{Retrieval-Augmented Generation.}
\cite{lewis2020rag} introduced RAG, which...

\paragraph{Hybrid Search.}
\cite{robertson2009bm25} proposed BM25...
\cite{cormack2009rrf} introduced Reciprocal Rank Fusion...

\paragraph{Temporal-Aware Retrieval.}
Prior work on temporal retrieval...

\paragraph{Query Expansion.}
\cite{ma2023hyde} proposed HyDE...
```

### Bước 4: Viết Section 3 — Problem Formulation

```latex
\section{Problem Formulation}

\textbf{Definition.} Given a large-scale document corpus 
$\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$ where $N > 500\text{k}$,
each document $d_i = (c_i, s_i, t_i)$ consists of content $c_i$,
source type $s_i \in \{\text{slack, gmail, jira, confluence}\}$,
and timestamp $t_i$.

Given a user query $q$, the goal is to retrieve a ranked list
$\mathcal{R} = [d_{r_1}, d_{r_2}, \ldots, d_{r_K}]$ that maximizes
retrieval quality metrics (MRR@K, Recall@K).
```

### Bước 5: Viết Section 1 — Introduction

```latex
\section{Introduction}

Enterprise organizations accumulate vast document corpora...
[Hook: 500k documents, multiple formats, high update frequency]

Standard RAG systems face three fundamental challenges at this scale:
(1) \textbf{Temporal blindness}...
(2) \textbf{Vector space density}...
(3) \textbf{Vocabulary mismatch}...

We propose \textbf{T-RAG} (Temporal and Targeted RAG), a system that
addresses these challenges through four key contributions:
\begin{enumerate}
    \item \textbf{Unified Storage...}
    \item \textbf{Self-Query Expansion...}
    \item \textbf{Hybrid Retrieval...}
    \item \textbf{Conditional Temporal Reranking...}
\end{enumerate}

Experiments on EnterpriseRAG-Bench demonstrate that T-RAG achieves
X\% improvement in MRR@10 over Standard RAG...
```

### Bước 6: Viết Abstract (CUỐI CÙNG)

```latex
\begin{abstract}
Retrieval-Augmented Generation (RAG) systems face significant challenges 
when deployed on large-scale enterprise document corpora (500k+ documents).
We identify three core failure modes: temporal blindness, vector space 
density degradation, and vocabulary mismatch in enterprise contexts.
We propose T-RAG (Temporal and Targeted RAG), a pipeline that integrates 
(1) unified LanceDB storage for vector, full-text, and metadata search; 
(2) LLM-based self-query expansion with temporal intent detection; 
(3) hybrid dense-sparse retrieval via Reciprocal Rank Fusion; and 
(4) conditional temporal reranking using exponential time decay. 
Experiments on EnterpriseRAG-Bench show that T-RAG achieves X\% improvement 
in MRR@10 over Standard RAG, with particular gains on temporal queries (Y\%).
\end{abstract}
```

---

## STEP 5.3 — Internal Review Process

### Round 1: Self-review checklist

- [ ] Mọi claim có citation hoặc số liệu?
- [ ] Mọi Figure/Table được reference trong text?
- [ ] Mọi ký hiệu toán học được định nghĩa lần đầu xuất hiện?
- [ ] Có Limitations section?
- [ ] Abstract đủ: motivation, method, result?
- [ ] Tổng số trang ≤ 8 (ACL format)?

### Round 2: Peer review (đồng tác giả Bằng)

- Đọc toàn bộ và comment trực tiếp vào LaTeX
- Dùng `\todo{}` command để đánh dấu chỗ cần sửa

```latex
% Trong preamble
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\note}[1]{\textcolor{blue}{[NOTE: #1]}}
```

### Round 3: English polish

- Paste từng section vào Grammarly Premium
- Hoặc dùng: `grammarly --document paper/main.tex`

---

## STEP 6.1 — Submission Checklist

### Trước submission

- [ ] Compile `main.tex` không có warning
- [ ] Chạy ACL formatting check tool
- [ ] Xóa toàn bộ `\todo{}` comments
- [ ] Remove author info (anonymous submission)
- [ ] Kiểm tra page count (≤ 8 pages content + unlimited references)
- [ ] Kiểm tra figure resolution (đã dùng PDF vector)
- [ ] Kiểm tra file size < 10MB

### Submit arXiv (TRƯỚC venue)

```bash
# Prepare arXiv submission
# 1. Add author info back
# 2. Zip: main.tex + figures/ + tables/ + acl_latex.sty + references.bib
zip -r trag_arxiv_submission.zip paper/

# Upload to: https://arxiv.org/submit
# Category: cs.IR (Information Retrieval) + cs.CL (Computation and Language)
```

### Submit to venue

1. Tạo account trên OpenReview / HotCRP (tùy venue)
2. Upload PDF (anonymous version)
3. Submit supplementary material (code link, data link)
4. Confirm submission trước deadline

### Sau submission

- [ ] Thông báo arXiv preprint trên Twitter/LinkedIn
- [ ] Upload code lên GitHub: `git tag v1.0.0 && git push origin v1.0.0`
- [ ] Cập nhật README với link arXiv

---

## ✅ Phase 5 & 6 Done Criteria

**Paper:**
- [ ] `paper/main.tex` compile được, không lỗi
- [ ] Tổng 8 trang nội dung (không tính references)
- [ ] Có đủ: Abstract, Introduction, Related Work, Methodology, Experiments, Conclusion, Limitations
- [ ] Table 1 (Main Results) + Table 2 (Ablation) hoàn chỉnh với số liệu thật
- [ ] Figure 1 (Architecture) + Figure 2 (Ablation bar chart) dạng PDF vector

**Submission:**
- [ ] arXiv preprint đã được public
- [ ] Đã submit đến ít nhất 1 venue
- [ ] GitHub repo public với README đầy đủ

## Estimated Time: 3 tuần (2 viết + 1 submit)
