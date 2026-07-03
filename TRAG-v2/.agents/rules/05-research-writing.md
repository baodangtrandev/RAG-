# Rule 05: Research & Paper Writing Standards

## Scope
Áp dụng khi viết bài báo khoa học, review literature, và trình bày kết quả thực nghiệm.

---

## 1. Nguyên tắc về claim & evidence

> **"No claim without citation or empirical evidence."**

| Loại claim | Yêu cầu |
|-----------|---------|
| "Standard RAG bị temporal blindness" | Phải dẫn paper hoặc ví dụ thực nghiệm |
| "BM25 tốt hơn cho keyword queries" | Phải có kết quả ablation (EXP-01 vs EXP-00) |
| "T-RAG đạt SOTA trên EnterpriseRAG-Bench" | Phải có Table kết quả so sánh đầy đủ |

---

## 2. Citation Management

**Tool:** Zotero + Better BibTeX

Tất cả references phải lưu trong `paper/references.bib`.

### Format bibtex chuẩn

```bibtex
@inproceedings{lewis2020rag,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

### Quy tắc citation key: `[firstauthor][year][keyword]`

- `lewis2020rag` — Lewis 2020, về RAG
- `robertson2009bm25` — Robertson 2009, về BM25
- `nogueira2019reranker` — Nogueira 2019, về reranker

---

## 3. Cấu trúc paper chuẩn (ACL format)

```
paper/
├── main.tex             # Main paper file
├── references.bib       # BibTeX references
├── figures/
│   ├── trag_architecture.pdf   # Fig 1: Kiến trúc pipeline
│   ├── ablation_bar.pdf        # Fig 2: Ablation results
│   └── time_decay_curve.pdf    # Fig 3: λ sensitivity
├── tables/
│   ├── main_results.tex        # Table 1: Main comparison
│   └── ablation_results.tex    # Table 2: Ablation study
└── acl_latex.sty               # ACL style file
```

---

## 4. Quy tắc về Figures & Tables

### Figures
- Định dạng: **PDF vector** (không dùng PNG cho paper).
- Tool tạo figure: **matplotlib** (export PDF) hoặc **TikZ** trong LaTeX.
- Mọi figure phải có caption đầy đủ, tự giải thích được.
- Font size trong figure ≥ font body text.

```python
# Xuất figure chuẩn cho paper
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "text.usetex": True,
})
fig.savefig("paper/figures/ablation_bar.pdf", bbox_inches="tight", dpi=300)
```

### Tables
- Dùng **booktabs** trong LaTeX (không dùng `\hline`).
- **Bold** best score trong mỗi column.
- Luôn có dòng `±std` hoặc confidence interval nếu có thể.

```latex
% Ví dụ Table chuẩn
\begin{table}[t]
\centering
\caption{Main results on EnterpriseRAG-Bench.}
\begin{tabular}{lrrr}
\toprule
\textbf{System} & \textbf{MRR@10} & \textbf{Recall@5} & \textbf{NDCG@10} \\
\midrule
Standard RAG    & 0.367 & 0.481 & 0.342 \\
BM25-only       & 0.341 & 0.453 & 0.318 \\
Dense-only      & 0.389 & 0.502 & 0.361 \\
\midrule
\textbf{T-RAG (Ours)} & \textbf{0.512} & \textbf{0.643} & \textbf{0.489} \\
\bottomrule
\end{tabular}
\label{tab:main_results}
\end{table}
```

---

## 5. Thứ tự viết paper (quan trọng!)

| Bước | Section | Lý do |
|------|---------|-------|
| 1 | Section 5 — Experiments | Có số liệu rồi → viết trước khi "nguội" |
| 2 | Section 3 — Methodology | Mô tả hệ thống đã implement |
| 3 | Section 4 — Problem Formulation | Formal hóa bài toán sau khi biết giải pháp |
| 4 | Section 2 — Related Work | Dựa trên notes lit review |
| 5 | Section 1 — Introduction | Biết kết quả mới viết intro thuyết phục |
| 6 | Abstract | Viết cuối cùng — tóm tắt toàn bộ |

---

## 6. Review process nội bộ

Trước khi submit phải qua **3 vòng review**:

| Vòng | Người review | Focus |
|------|-------------|-------|
| Self-review | Tác giả | Logic, flow, completeness |
| Peer-review | Đồng tác giả | Technical correctness, clarity |
| English polish | Native speaker / Tool | Grammar, academic tone |

### Checklist tự review

- [ ] Mỗi section có thể đọc độc lập không?
- [ ] Figure và Table có được reference trong text không?
- [ ] Mọi ký hiệu toán học có được định nghĩa không?
- [ ] Có Limitations section không?
- [ ] Abstract có đủ: motivation, method, result, conclusion?

---

## 7. Target venues và deadlines

| Venue | Deadline submit | Announce | Notes |
|-------|----------------|---------|-------|
| **arXiv** | Rolling | Immediate | Submit ngay sau Phase 5 |
| **ECIR 2027** | ~Oct 2026 | ~Dec 2026 | First-choice conference |
| **SIGIR 2027** | ~Jan 2027 | ~Apr 2027 | Top venue cho Information Retrieval |
| **EMNLP 2026** | ~Jun 2026 | ~Sep 2026 | NLP venue nếu kịp deadline |

> [!IMPORTANT]
> Submit lên **arXiv trước** bất kỳ venue nào để establish timestamp và priority.
