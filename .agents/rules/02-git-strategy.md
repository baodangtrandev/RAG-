# Rule 02: Git & Branch Strategy

## Scope
Áp dụng cho toàn bộ version control workflow của dự án T-RAG.

---

## 1. Branch Strategy

```
main
 └── dev
      ├── feat/phase1-lit-review
      ├── feat/phase2-data-ingestion
      ├── feat/phase3-query-parser
      ├── feat/phase3-hybrid-retriever
      ├── feat/phase3-temporal-reranker
      ├── exp/exp-00-baseline
      ├── exp/exp-01-hybrid-only
      ├── exp/exp-05-full-trag
      └── fix/ingestion-crash-500k
```

### Quy tắc naming branch

| Prefix | Dùng khi | Ví dụ |
|--------|---------|-------|
| `feat/` | Implement tính năng mới | `feat/phase3-rrf-fusion` |
| `exp/` | Chạy thử nghiệm / experiment | `exp/exp-03-time-decay` |
| `fix/` | Sửa bug | `fix/lance-connection-timeout` |
| `docs/` | Cập nhật tài liệu | `docs/update-readme` |
| `refactor/` | Tái cấu trúc code | `refactor/chunker-interface` |

---

## 2. Commit Message Convention

Format: **Conventional Commits**

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

### Types

| Type | Dùng khi |
|------|---------|
| `feat` | Thêm tính năng mới |
| `fix` | Sửa bug |
| `exp` | Thêm / cập nhật experiment |
| `test` | Thêm / sửa test |
| `docs` | Cập nhật tài liệu |
| `refactor` | Refactor code, không đổi behavior |
| `chore` | Cập nhật dependencies, config |
| `perf` | Cải thiện performance |

### Ví dụ commit tốt

```
feat(retriever): implement RRF fusion for hybrid search

- Add reciprocal_rank_fusion() function
- Support configurable k parameter (default=60)
- Unit test: tests/test_rrf_fusion.py

Closes #12
```

```
exp(ablation): run EXP-01 hybrid-only on EnterpriseRAG-Bench

Results:
- MRR@10: 0.412 (vs baseline 0.367)
- Recall@5: 0.538 (vs baseline 0.481)

Config: configs/exp/exp-01.yaml
```

### ❌ Commit tệ (không được phép)

```
fix bug
update code
test
wip
```

---

## 3. Pull Request Process

### Khi nào mở PR?
- Khi một feature/experiment trên branch **đã hoàn thành** và cần merge vào `dev`.
- **Không** merge trực tiếp vào `main` (chỉ merge khi paper submitted).

### PR Template

```markdown
## Mô tả thay đổi
<!-- Giải thích ngắn gọn PR này làm gì -->

## Loại thay đổi
- [ ] Feature mới
- [ ] Bug fix
- [ ] Experiment
- [ ] Refactor

## Checklist
- [ ] Code pass `ruff`, `black`, `mypy`
- [ ] Tất cả tests pass (`pytest tests/ -v`)
- [ ] Đã cập nhật docs/config nếu cần
- [ ] Kết quả experiment đã được log vào `results/`

## Kết quả (nếu là experiment)
| Metric | Baseline | PR này |
|--------|---------|--------|
| MRR@10 | | |
| Recall@5 | | |
```

---

## 4. Quy tắc merge

- **Squash merge** khi merge `feat/*` → `dev` (giữ history clean).
- **Merge commit** khi merge `dev` → `main` (để preserve milestone).
- **Không** force push vào `dev` hoặc `main`.
- **Không** commit trực tiếp vào `main`.

---

## 5. Tagging & Versioning

| Tag | Thời điểm | Ví dụ |
|-----|----------|-------|
| `v0.1.0` | Sau khi baseline chạy xong | Phase 2 done |
| `v0.2.0` | Sau khi T-RAG full pipeline done | Phase 3 done |
| `v1.0.0` | Khi submit paper | Phase 6 |

```bash
git tag -a v0.1.0 -m "Phase 2: Baseline evaluation complete"
git push origin v0.1.0
```

---

## ✅ Tóm tắt nhanh

```bash
# Bắt đầu task mới
git checkout dev
git pull origin dev
git checkout -b feat/phase3-query-parser

# Commit đúng chuẩn
git add src/query_parser/expander.py tests/test_expander.py
git commit -m "feat(query-parser): implement LLM-based self-query expansion"

# Push và mở PR
git push origin feat/phase3-query-parser
# → Mở PR trên GitHub: feat/phase3-query-parser → dev
```
