"""
Cross-Encoder Reranker Module (Dynamic Thresholding)
=====================================================
Danh gia lai diem lien quan giua cau hoi va tai lieu bang Cross-Encoder.
Loc bo cac tai lieu co diem duoi nguong RERANKER_THRESHOLD.

Input:  List[str] queries, List[List[dict]] docs_per_query
Output: List[dict] — moi entry co "docs" (da rerank) va "is_unanswerable" flag
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)


def _load_env_float(key: str, default: float) -> float:
    """Doc float tu env voi fallback an toan."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning(
            "[Reranker] Gia tri '%s' cua %s khong hop le. Dung mac dinh: %s",
            os.environ.get(key),
            key,
            default,
        )
        return default


class CrossEncoderReranker:
    """Cross-Encoder Reranker voi kha nang batch inference tren GPU."""

    def __init__(self, model_name: Optional[str] = None, threshold: Optional[float] = None):
        self.model_name = model_name or os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.threshold = threshold if threshold is not None else _load_env_float("RERANKER_THRESHOLD", 0.0)

        logger.info(
            "[Reranker] Khoi tao CrossEncoder: model='%s', threshold=%s",
            self.model_name,
            self.threshold,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(self.model_name, device=device)

        logger.info("[Reranker] Model loaded on %s", device.upper())

    def rerank_batch(
        self,
        queries: List[str],
        docs_per_query: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank tat ca queries va docs trong mot lan batch inference.

        Args:
            queries:        N cau hoi.
            docs_per_query: N danh sach tai lieu (moi doc la dict co key "content").

        Returns:
            List[dict] dai N, moi entry gom:
                "docs"            : List[dict] sorted theo rerank_score giam dan
                "is_unanswerable" : bool -- True neu khong con doc nao sau loc
        """
        assert len(queries) == len(docs_per_query), "[Reranker] Mismatch: %d queries nhung %d doc lists" % (
            len(queries),
            len(docs_per_query),
        )

        n_queries = len(queries)
        total_pairs = sum(len(d) for d in docs_per_query)

        logger.info(
            "[Reranker] INPUT: %d queries | %d pairs (query, doc) | threshold=%s",
            n_queries,
            total_pairs,
            self.threshold,
        )

        if total_pairs == 0:
            logger.warning("[Reranker] No docs to rerank -> all unanswerable.")
            return [{"docs": [], "is_unanswerable": True} for _ in queries]

        # --- Step 1: Create flat list of all (query, content) pairs ---
        flat_pairs: List[Tuple[str, str]] = []
        pair_index: List[Tuple[int, int]] = []  # (query_idx, doc_idx_within_query)

        for q_idx, (query, docs) in enumerate(zip(queries, docs_per_query)):
            for d_idx, doc in enumerate(docs):
                content = doc.get("content", "")
                flat_pairs.append((query, content))
                pair_index.append((q_idx, d_idx))

        # --- Step 2: Single batch inference on GPU ---
        t0 = time.perf_counter()
        scores = self.model.predict(flat_pairs, batch_size=256, show_progress_bar=False)
        elapsed = time.perf_counter() - t0

        logger.info(
            "[Reranker] Inference done: %d pairs | %.2fs | %.0f pairs/s",
            total_pairs,
            elapsed,
            total_pairs / max(elapsed, 1e-9),
        )

        # --- Step 3: Map scores back to each doc ---
        score_matrix: List[List[float]] = [[float("-inf")] * len(docs) for docs in docs_per_query]
        for flat_idx, (q_idx, d_idx) in enumerate(pair_index):
            score_matrix[q_idx][d_idx] = float(scores[flat_idx])

        # --- Step 4: Filter by threshold and sort descending ---
        results = []
        n_dropped_total = 0

        for q_idx, (docs, scores_q) in enumerate(zip(docs_per_query, score_matrix)):
            kept = []
            dropped = 0
            for doc, score in zip(docs, scores_q):
                if score >= self.threshold:
                    kept.append({**doc, "rerank_score": score})
                else:
                    dropped += 1

            n_dropped_total += dropped
            kept.sort(key=lambda x: x["rerank_score"], reverse=True)

            is_unanswerable = len(kept) == 0
            results.append({"docs": kept, "is_unanswerable": is_unanswerable})

            logger.debug(
                "[Reranker] Query[%d]: kept=%d, dropped=%d, unanswerable=%s",
                q_idx,
                len(kept),
                dropped,
                is_unanswerable,
            )

        n_unanswerable = sum(1 for r in results if r["is_unanswerable"])
        logger.info(
            "[Reranker] OUTPUT: %d/%d unanswerable | total_dropped=%d docs",
            n_unanswerable,
            n_queries,
            n_dropped_total,
        )

        return results
