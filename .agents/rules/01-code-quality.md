# Rule 01: Code Quality Standards

## Scope
Áp dụng cho tất cả Python code trong `src/`, `tests/`, `scripts/`.

---

## 1. Toolchain bắt buộc

| Tool | Mục đích | Config file |
|------|---------|-------------|
| `ruff` | Linting + import sorting | `pyproject.toml` |
| `black` | Code formatting | `pyproject.toml` |
| `mypy` | Static type checking | `mypy.ini` |
| `pytest` | Unit & integration tests | `pytest.ini` |

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.mypy]
strict = true
python_version = "3.10"
```

---

## 2. Quy tắc đặt tên

| Element | Convention | Ví dụ |
|---------|-----------|-------|
| Module | `snake_case` | `temporal_reranker.py` |
| Class | `PascalCase` | `HybridRetriever` |
| Function | `snake_case` | `compute_rrf_score()` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_TOP_K = 50` |
| Config key | `snake_case` | `lambda_decay: 0.01` |

---

## 3. Type Hints — Bắt buộc 100%

```python
# ✅ Đúng
def compute_temporal_score(
    relevance: float,
    delta_t_days: float,
    lambda_decay: float,
) -> float:
    return relevance * math.exp(-lambda_decay * delta_t_days)

# ❌ Sai — không có type hints
def compute_temporal_score(relevance, delta_t_days, lambda_decay):
    return relevance * math.exp(-lambda_decay * delta_t_days)
```

---

## 4. Docstring — Bắt buộc cho mọi public function/class

Format: **Google Style Docstring**

```python
def hybrid_retrieve(
    query: str,
    top_k: int = 50,
    source_filter: str | None = None,
) -> list[Document]:
    """Retrieve documents using hybrid dense + sparse search.

    Args:
        query: Raw user query string.
        top_k: Number of candidate documents to return.
        source_filter: Optional source type filter (e.g., "gmail", "jira").

    Returns:
        List of Document objects ranked by RRF score.

    Raises:
        LanceDBConnectionError: If the database connection is unavailable.
    """
```

---

## 5. Không hardcode — dùng Config

```python
# ❌ Sai
results = retriever.search(query, top_k=50)

# ✅ Đúng — load từ YAML config
from src.config import RetrieverConfig
cfg = RetrieverConfig.from_yaml("configs/retriever.yaml")
results = retriever.search(query, top_k=cfg.top_k)
```

---

## 6. Error Handling

```python
# ✅ Đúng — log lỗi cụ thể, raise exception có context
import logging
logger = logging.getLogger(__name__)

try:
    results = db.search(query_vector)
except Exception as e:
    logger.error("LanceDB search failed for query='%s': %s", query, e)
    raise RuntimeError(f"Retrieval failed: {e}") from e
```

---

## 7. Pre-commit Hook

Cài đặt bắt buộc trước khi commit code lần đầu:

```bash
pip install pre-commit
pre-commit install
```

File `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
```

---

## ✅ Checklist trước mỗi PR

- [ ] `ruff check src/ tests/` — không có lỗi
- [ ] `black --check src/ tests/` — không có diff
- [ ] `mypy src/` — không có type error
- [ ] `pytest tests/ -v` — tất cả test pass
- [ ] Không có hardcoded values
- [ ] Mọi public function đều có docstring
