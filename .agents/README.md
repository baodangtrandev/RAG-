# .agents — T-RAG Project Agent Configuration

Thư mục này chứa toàn bộ rules và workflows cho dự án T-RAG.
Mọi AI agent làm việc trên dự án này **phải đọc các file trong thư mục này trước**.

---

## 📁 Cấu trúc

```
.agents/
├── skills/                    # Installed AI skills
│   ├── rag-engineer/          # RAG engineering expertise
│   └── find-skills/           # Skill discovery
│
├── rules/                     # Quy tắc dự án (BẮT BUỘC tuân thủ)
│   ├── 01-code-quality.md     # Coding standards (ruff, black, mypy, type hints)
│   ├── 02-git-strategy.md     # Branch naming, commit convention, PR process
│   ├── 03-experiment-tracking.md  # Reproducibility & experiment logging
│   ├── 04-project-structure.md    # Directory layout & module interfaces
│   └── 05-research-writing.md     # Paper writing standards & LaTeX conventions
│
└── workflows/                 # Quy trình từng phase (Step-by-step)
    ├── phase1-lit-review.md       # Literature review & problem formulation
    ├── phase2-data-baseline.md    # Data setup & baseline evaluation
    ├── phase3-implementation.md   # T-RAG core implementation (4 modules)
    ├── phase4-experiments.md      # Ablation study (6 experiments)
    └── phase5-6-paper-submission.md  # Paper writing & venue submission
```

---

## 🚀 Quick Start cho Agent mới

1. **Đọc rules** theo thứ tự 01 → 05
2. **Xác định phase hiện tại** dựa trên trạng thái project
3. **Đọc workflow tương ứng** và làm theo step-by-step
4. **Áp dụng skill** `rag-engineer` cho mọi quyết định kỹ thuật về RAG

---

## 📋 Project Summary

**Tên dự án:** T-RAG (Temporal & Targeted RAG)

**Idea:**
- Standard RAG fail trên large-scale enterprise data (500k+ docs) vì:
  1. Temporal Blindness — không ưu tiên tài liệu mới
  2. Vector Density — khó phân biệt docs trong không gian cao chiều
  3. Vocabulary Mismatch — query ngôn ngữ tự nhiên vs. docs chứa ID/jargon
- T-RAG giải quyết bằng: Metadata Filtering + Hybrid Search + Time Decay + Query Expansion

**Dataset:** EnterpriseRAG-Bench (500k docs: Slack, Gmail, Jira, Confluence)

**Tech stack:** LanceDB, vLLM, BGE embeddings, BGE Reranker, RAGAS

**Target venue:** ECIR 2027 / SIGIR 2027 / arXiv (preprint)

---

## ⚡ Trạng thái hiện tại

| Phase | Status | Note |
|-------|--------|------|
| Phase 1 — Literature Review | 🔲 Pending | Bắt đầu từ đây |
| Phase 2 — Data & Baseline | 🔲 Pending | — |
| Phase 3 — Implementation | 🔲 Pending | — |
| Phase 4 — Experiments | 🔲 Pending | — |
| Phase 5 — Paper Writing | 🔲 Pending | — |
| Phase 6 — Submission | 🔲 Pending | — |

> Cập nhật bảng này sau mỗi phase hoàn thành.
