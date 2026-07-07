# Workflow: Phase 1 — Problem Formulation & Literature Review

## Trigger
Bắt đầu dự án hoặc khi cần xác định lại research direction.

## Goal
Hoàn thành research proposal học thuật, xác định baselines, và lên danh sách 20+ related papers tập trung vào Semantic Routing, Hybrid Search, và Multi-hop Reasoning.

---

## Step-by-step Workflow

### STEP 1.1 — Đọc EnterpriseRAG-Bench Paper

**Action:**
- Tìm hiểu dataset EnterpriseRAG-Bench với 511k documents.

**Output:** File `docs/lit_review/enterprise_rag_bench_notes.md` (Đã cập nhật sau EDA: Không có timestamp).

---

### STEP 1.2 — Literature Review (20+ papers)

**Search queries để dùng:**
```
"Semantic routing" "RAG" site:arxiv.org
"Multi-hop reasoning" "Retrieval-Augmented Generation"
"Hybrid search BM25 dense retrieval"
"Cross-encoder reranker"
"Reciprocal rank fusion"
```

**Target papers cần đọc (mandatory):**

| # | Paper | Relevance |
|---|-------|---------|
| 1 | Qiao et al. (2024) — Route Before Retrieve | Semantic Routing |
| 2 | Wang et al. (2024) — RAGRouter | Semantic Routing |
| 3 | Cormack et al. (2009) — Reciprocal Rank Fusion | RRF (Foundation for SW-RRF) |
| 4 | Tang et al. (2024) — MultiHop-RAG | Multi-hop reasoning (CSEP) |
| 5 | Zhang et al. (2025) — HopRAG | Multi-hop reasoning |
| 6 | Nogueira et al. (2019) — Passage Re-ranking with BERT | Reranker |
| 7 | Lewis et al. (2020) — RAG original | Baseline |
| 8 | Sun et al. (2026) — EnterpriseRAG-Bench | Dataset |

**Output:** File `docs/lit_review/literature_review.md`

---

### STEP 1.3 — Viết Research Gap Document

**Output:** File `docs/research_gap.md`

- **Khoảng trống 1:** Mật độ Không gian Vector & Sự nhập nhằng nguồn (Vector Density + Source Ambiguity).
- **Khoảng trống 2:** Bất đồng từ vựng trong văn cảnh doanh nghiệp (Vocabulary Mismatch).
- **Khoảng trống 3:** Suy giảm hiệu suất ở Multi-Source Retrieval (Cross-Source Context).

---

### STEP 1.4 — Viết Research Proposal

**Output:** File `docs/proposal.md` (Đã hoàn thành)

**Đóng góp cốt lõi:**
1. **Probabilistic Source Router (PSR):** Giảm search space từ 50-90%.
2. **Source-Weighted RRF (SW-RRF):** Tích hợp Bayesian Prior vào RRF.
3. **Cross-Source Entity Propagation (CSEP):** Multi-hop retrieval cho đa nguồn.

---

### STEP 1.5 — Setup tracking bảng phác thảo Table 1

Tạo file `docs/table1_draft.md`:

```markdown
# Table 1 (Draft — No numbers yet)

| System              | Recall@10 | NDCG@10 | Search Space | Latency P50 |
|---------------------|-----------|---------|--------------|-------------|
| Standard RAG        | —         | —       | 100% (511k)  | —           |
| + Metadata Filter   | —         | —       | ~50%         | —           |
| + SW-RRF            | —         | —       | ~30%         | —           |
| **T-RAG (Full)**    | **—**     | **—**   | **~10-20%**  | **—**       |
```

---

## ✅ Phase 1 Done Criteria

- [x] `docs/lit_review/enterprise_rag_bench_notes.md` — Hoàn thành (Cập nhật sau EDA)
- [x] `docs/lit_review/literature_review.md` — Hoàn thành (Hướng Semantic Routing)
- [x] `docs/research_gap.md` — Hoàn thành (Dựa trên 3 điểm yếu cốt lõi)
- [x] `docs/proposal.md` — Hoàn thành (Academic format, có công thức)
- [x] `docs/table1_draft.md` — Đã setup bảng

## Estimated Time: 2 tuần
