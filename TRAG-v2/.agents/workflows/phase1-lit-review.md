# Workflow: Phase 1 — Problem Formulation & Literature Review

## Trigger
Bắt đầu dự án hoặc khi cần xác định lại research direction.

## Goal
Hoàn thành research proposal 2 trang, xác định baselines, và lên danh sách 20+ related papers.

---

## Step-by-step Workflow

### STEP 1.1 — Đọc EnterpriseRAG-Bench Paper

**Action:**
```bash
# Tìm paper gốc
# Search: "EnterpriseRAG-Bench" on arXiv or Semantic Scholar
# URL: https://arxiv.org/search/?searchtype=all&query=EnterpriseRAG-Bench
```

**Output:** File `docs/lit_review/enterprise_rag_bench_notes.md`

**Template notes:**
```markdown
## EnterpriseRAG-Bench — Reading Notes

**Paper:** [Title, Authors, Year]
**Link:** [arXiv URL]

### Dataset Characteristics
- Scale: 500k+ documents
- Sources: Slack (285k), Gmail (121k), Jira (41k), Confluence (5k)
- ...

### Evaluation Protocol
- Metrics: [list metrics used]
- Split: [train/val/test details]

### Key Findings
1. ...

### Limitations mentioned by authors
1. ...

### Research Gap (our angle)
- Standard RAG fails because: ...
```

---

### STEP 1.2 — Literature Review (20+ papers)

**Search queries để dùng:**
```
"RAG survey" site:arxiv.org
"temporal-aware retrieval" site:arxiv.org
"hybrid search BM25 dense retrieval"
"query expansion LLM"
"reciprocal rank fusion"
"time decay document retrieval"
"cross-encoder reranker"
"HyDE hypothetical document embedding"
```

**Target papers cần đọc (mandatory):**

| # | Paper | Relevance |
|---|-------|---------|
| 1 | Lewis et al. (2020) — RAG original | Core |
| 2 | Gao et al. (2023) — RAG survey | Core |
| 3 | Robertson et al. (2009) — BM25 | Sparse retrieval |
| 4 | Nogueira et al. (2019) — mono/duo BERT reranker | Reranker |
| 5 | Ma et al. (2023) — HyDE | Query expansion |
| 6 | Cormack et al. (2009) — Reciprocal Rank Fusion | RRF |
| 7 | Shi et al. (2023) — REPLUG | Retrieval-LLM integration |
| 8 | Asai et al. (2023) — Self-RAG | Adaptive RAG |
| 9 | Adlakha et al. (2023) — EnterpriseRAG-Bench | Dataset |
| 10+ | Papers liên quan đến temporal search | Temporal |

**Output:** File `docs/lit_review/literature_review.md`

---

### STEP 1.3 — Viết Research Gap Document

**Output:** File `docs/research_gap.md`

**Template:**
```markdown
# Research Gap Analysis

## Problem Statement
Retrieval-Augmented Generation (RAG) systems face significant challenges
when deployed on large-scale enterprise datasets (500k+ documents).

## Identified Gaps

### Gap 1: Temporal Blindness
- **Symptom:** Standard RAG returns outdated information for time-sensitive queries
- **Root cause:** Cosine similarity ignores document timestamps
- **Our solution:** Conditional Time Decay reranking with λ parameter

### Gap 2: Vector Space Density at Scale
- **Symptom:** At 500k documents, cosine distances cluster → poor discrimination
- **Root cause:** Standard RAG searches entire vector space
- **Our solution:** Metadata pre-filtering reduces search space before embedding search

### Gap 3: Vocabulary Mismatch in Enterprise Contexts
- **Symptom:** Queries use natural language; docs contain IDs, ticket numbers, jargon
- **Root cause:** Dense-only search misses exact term matches
- **Our solution:** Hybrid search (Dense + BM25/Tantivy) via RRF

## T-RAG Contributions
1. **Unified Storage (LanceDB):** Single DB for vector, FTS, and metadata
2. **Self-Query Expansion:** LLM-based query understanding + HyDE
3. **Hybrid Retrieval:** Dense + Sparse with RRF fusion
4. **Conditional Temporal Reranking:** Time decay only when required
```

---

### STEP 1.4 — Viết Research Proposal

**Output:** File `docs/research_proposal.md` (2 trang)

**Sections bắt buộc:**
1. **Title:** T-RAG: Temporal and Targeted Retrieval-Augmented Generation for Large-Scale Enterprise Documents
2. **Research Questions (RQ):**
   - RQ1: Can temporal-aware reranking improve retrieval for time-sensitive queries?
   - RQ2: Does metadata pre-filtering improve both precision and latency at 500k scale?
   - RQ3: How much does each component contribute individually? (ablation)
3. **Hypothesis:** T-RAG will outperform Standard RAG on MRR@10 by >10% on temporal query subsets
4. **Planned Baselines:** Standard RAG, BM25-only, Dense-only, Hybrid-only
5. **Proposed Metrics:** MRR@10, Recall@5, NDCG@10, Latency P50/P95
6. **Timeline estimate**

---

### STEP 1.5 — Setup tracking bảng phác thảo Table 1

Tạo file `docs/table1_draft.md`:

```markdown
# Table 1 (Draft — No numbers yet)

| System              | MRR@10 | Recall@5 | NDCG@10 | Latency P50 |
|---------------------|--------|---------|---------|-------------|
| BM25-only           | —      | —       | —       | —           |
| Dense-only (DPR)    | —      | —       | —       | —           |
| Standard RAG        | —      | —       | —       | —           |
| T-RAG (Ours)        | **—**  | **—**   | **—**   | —           |
```

---

## ✅ Phase 1 Done Criteria

- [ ] `docs/lit_review/literature_review.md` — ≥ 20 papers với notes
- [ ] `docs/research_gap.md` — 3 gaps được phân tích rõ
- [ ] `docs/research_proposal.md` — 2 trang, đủ RQ + hypothesis
- [ ] `docs/table1_draft.md` — Có cấu trúc bảng (chưa có số)
- [ ] Xác định được ít nhất 3 baselines
- [ ] Identify được target venue đầu tiên và deadline

## Estimated Time: 2 tuần
