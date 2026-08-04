import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Any, Dict, List, Tuple

import lancedb
import numpy as np
from dotenv import load_dotenv

from src.models.router_inference import ProbabilisticSourceRouter

load_dotenv()

logger = logging.getLogger(__name__)


class EnterpriseRetrieverV2:
    """EnterpriseRetrieverV2 implementing adaptive tau and SW-RRF retrieval."""

    def __init__(
        self,
        db_uri: str = None,
        model_dir: str = None,
        tau_base: float = 0.15,
        tau_alpha: float = 0.08,
        adaptive_tau: bool = True,
        gamma: float = 0.5,
        k_rrf: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.db_uri = db_uri or os.environ.get("RAG_DB_URI", "data/lancedb")
        model_dir = model_dir or os.environ.get("PSR_MODEL_DIR", "models/psr_v2")

        self.tau_base = tau_base
        self.tau_alpha = tau_alpha
        self.adaptive_tau = adaptive_tau
        self.gamma = gamma
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        logger.info(
            f"[T-RAG v2] Init: db={self.db_uri}, tau_base={self.tau_base}, tau_alpha={self.tau_alpha}, gamma={self.gamma}"
        )
        self.db = lancedb.connect(self.db_uri)
        self.router = ProbabilisticSourceRouter(model_dir=model_dir)

        self.table_names = self.db.table_names()
        self.tables = {name: self.db.open_table(name) for name in self.table_names}

        try:
            self.hybrid_search = os.environ.get("RAG_HYBRID_SEARCH", "True").lower() == "true"
        except Exception:
            self.hybrid_search = True

        self.table_sizes = {}
        for name, table in self.tables.items():
            try:
                self.table_sizes[name] = len(table)
            except Exception:
                self.table_sizes[name] = 0

    def compute_adaptive_tau(self, probs: np.ndarray) -> float:
        probs_sum = probs.sum()
        if probs_sum > 0:
            norm_probs = probs / probs_sum
            entropy = -np.sum(norm_probs * np.log(norm_probs + 1e-10))
        else:
            entropy = 0.0

        H_max = np.log(len(probs))
        confidence = 1.0 - (entropy / H_max)
        tau_eff = self.tau_base + self.tau_alpha * (confidence - 0.5)
        return float(max(0.05, min(0.40, tau_eff)))

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, float], np.ndarray]:
        emb = self.router.encoder.encode([query], normalize_embeddings=True)
        probs = self.router.clf.predict_proba(emb)[0]

        source_probs = {self.router.classes[i]: float(probs[i]) for i in range(len(probs))}

        if self.adaptive_tau:
            tau_eff = self.compute_adaptive_tau(probs)
        else:
            tau_eff = self.tau_base

        active_shards = [s for s, p in source_probs.items() if p >= tau_eff]
        if not active_shards:
            best_source = max(source_probs, key=source_probs.get)
            active_shards.append(best_source)

        logger.debug(f"[Router v2] tau_eff={tau_eff:.4f} | Quét {len(active_shards)}/9 bảng -> {active_shards}")

        search_space_docs = sum(self.table_sizes.get(source, 0) for source in active_shards)
        search_limit = max(top_k * 2, 10)
        dense_candidates = []
        sparse_candidates = []

        for source in active_shards:
            if source not in self.tables:
                continue

            table = self.tables[source]
            p_s = source_probs[source]
            prior_weight = p_s**self.gamma

            try:
                results = table.search(emb[0]).limit(search_limit).to_list()
                for doc in results:
                    doc["_source"] = source
                    doc["_prior_weight"] = prior_weight
                    doc["_router_prob"] = p_s
                    dense_candidates.append(doc)
            except Exception as e:
                logger.error(f"Vector search error in table {source}: {e}")

            if self.hybrid_search:
                try:
                    results_fts = table.search(query, query_type="fts").limit(search_limit).to_list()
                    for doc in results_fts:
                        doc["_source"] = source
                        doc["_prior_weight"] = prior_weight
                        doc["_router_prob"] = p_s
                        sparse_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"FTS search error in table {source}: {e}")

        fused_docs = {}

        dense_candidates.sort(key=lambda x: x.get("_distance", float("inf")))
        for rank_0_idx, doc in enumerate(dense_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": rank_0_idx + 1, "sparse_rank": None}
            else:
                fused_docs[key]["dense_rank"] = rank_0_idx + 1

        sparse_candidates.sort(key=lambda x: x.get("_score", x.get("score", 0.0)), reverse=True)
        for rank_0_idx, doc in enumerate(sparse_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": None, "sparse_rank": rank_0_idx + 1}
            else:
                fused_docs[key]["sparse_rank"] = rank_0_idx + 1

        all_results = []
        for key, info in fused_docs.items():
            doc = info["doc"]
            dense_rank = info["dense_rank"]
            sparse_rank = info["sparse_rank"]

            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += self.dense_weight * (1.0 / (self.k_rrf + dense_rank))
            if sparse_rank is not None:
                rrf_score += self.sparse_weight * (1.0 / (self.k_rrf + sparse_rank))

            sw_rrf_score = doc["_prior_weight"] * rrf_score

            clean_doc = {
                "source": doc["_source"],
                "doc_id": doc.get("doc_id", "unknown"),
                "content": doc.get("content", ""),
                "title": doc.get("title", ""),
                "vector_distance": doc.get("_distance", 1.0),
                "router_prob": doc["_router_prob"],
                "sw_rrf_score": sw_rrf_score,
                "search_space_docs": search_space_docs,
            }
            all_results.append(clean_doc)

        all_results.sort(key=lambda x: x["sw_rrf_score"], reverse=True)
        final_top_k = all_results[:top_k]

        return final_top_k, source_probs, emb[0]
