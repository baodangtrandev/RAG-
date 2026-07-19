import os
import sys

# Thêm thư mục gốc vào sys.path để fix lỗi ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lancedb
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from src.models.router_inference import ProbabilisticSourceRouter

logger = logging.getLogger(__name__)

class EnterpriseRetrieverV2:
    """
    Retriever Pipeline tối ưu hóa (T-RAG v2)
    Bao gồm:
    1. Adaptive Tau (Entropy-based dynamic threshold)
    2. SW-RRF với tối ưu hóa tham số
    3. Fix double-encode trong luồng CSEP bằng cách trả về source_probs và embeddings
    """
    def __init__(self, db_uri: str = None, model_dir: str = None, 
                 tau_base: float = 0.15, tau_alpha: float = 0.08, adaptive_tau: bool = True,
                 gamma: float = 0.5, k_rrf: int = 60, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.db_uri = db_uri or os.environ.get("RAG_DB_URI", "data/lancedb")
        model_dir = model_dir or os.environ.get("PSR_MODEL_DIR", "models/psr_v2")
        
        self.tau_base = tau_base
        self.tau_alpha = tau_alpha
        self.adaptive_tau = adaptive_tau
        self.gamma = gamma
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(f"[T-RAG v2] Init: db={self.db_uri}, tau_base={self.tau_base}, tau_alpha={self.tau_alpha}, gamma={self.gamma}")
        self.db = lancedb.connect(self.db_uri)
        
        self.router = ProbabilisticSourceRouter(model_dir=model_dir)
        
        # Cache tables to avoid disk I/O bottleneck
        self.table_names = self.db.table_names()
        self.tables = {name: self.db.open_table(name) for name in self.table_names}
        
        try:
            self.hybrid_search = os.environ.get("RAG_HYBRID_SEARCH", "True").lower() == "true"
        except Exception:
            self.hybrid_search = True
            
        # Get document counts for search space calculation
        self.table_sizes = {}
        for name, table in self.tables.items():
            try:
                self.table_sizes[name] = len(table)
            except Exception:
                self.table_sizes[name] = 0

    def compute_adaptive_tau(self, probs: np.ndarray) -> float:
        """
        Tính dynamic tau dựa trên Shannon entropy của phân phối xác suất router
        """
        probs_sum = probs.sum()
        if probs_sum > 0:
            norm_probs = probs / probs_sum
            entropy = -np.sum(norm_probs * np.log(norm_probs + 1e-10))
        else:
            entropy = 0.0
            
        # H_max = ln(K) = ln(9) ≈ 2.197
        H_max = np.log(len(probs))
        confidence = 1.0 - (entropy / H_max)
        
        # tau_eff = tau_base + alpha * (confidence - 0.5)
        tau_eff = self.tau_base + self.tau_alpha * (confidence - 0.5)
        
        # Clamp tau_eff in [0.05, 0.40]
        return float(max(0.05, min(0.40, tau_eff)))

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, float], np.ndarray]:
        """
        Tìm kiếm tài liệu và trả về kèm source_probs, query embedding để tái sử dụng
        """
        # Bước 1: Mã hóa Vector và tính Xác suất qua Router
        emb = self.router.encoder.encode([query], normalize_embeddings=True)
        probs = self.router.clf.predict_proba(emb)[0]
        
        # Mapping Probability với Từng Bảng
        source_probs = {self.router.classes[i]: float(probs[i]) for i in range(len(probs))}
        
        # Bước 2: Tính dynamic tau
        if self.adaptive_tau:
            tau_eff = self.compute_adaptive_tau(probs)
        else:
            tau_eff = self.tau_base
            
        # Bước 3: Kích hoạt Shards
        active_shards = [s for s, p in source_probs.items() if p >= tau_eff]
        if not active_shards:
            best_source = max(source_probs, key=source_probs.get)
            active_shards.append(best_source)
            
        logger.debug(f"[Router v2] tau_eff={tau_eff:.4f} | Quét {len(active_shards)}/9 bảng -> {active_shards}")
        
        search_space_docs = sum(self.table_sizes.get(source, 0) for source in active_shards)
        search_limit = max(top_k * 2, 10)
        dense_candidates = []
        sparse_candidates = []
        
        # Bước 4: Tìm kiếm song song Dense & Sparse cục bộ
        for source in active_shards:
            if source not in self.tables:
                continue
                
            table = self.tables[source]
            p_s = source_probs[source]
            prior_weight = p_s ** self.gamma
            
            # Vector Search
            try:
                results = table.search(emb[0]).limit(search_limit).to_list()
                for doc in results:
                    doc["_source"] = source
                    doc["_prior_weight"] = prior_weight
                    doc["_router_prob"] = p_s
                    dense_candidates.append(doc)
            except Exception as e:
                logger.error(f"Lỗi vector search bảng {source}: {e}")
                
            # FTS Search
            if self.hybrid_search:
                try:
                    results_fts = table.search(query, query_type="fts").limit(search_limit).to_list()
                    for doc in results_fts:
                        doc["_source"] = source
                        doc["_prior_weight"] = prior_weight
                        doc["_router_prob"] = p_s
                        sparse_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Lỗi FTS search bảng {source}: {e}")
                    
        # Bước 5: SW-RRF Fusion
        fused_docs = {}
        
        # Sắp xếp Dense Candidates toàn cục theo distance
        dense_candidates.sort(key=lambda x: x.get("_distance", float('inf')))
        for rank_0_idx, doc in enumerate(dense_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {
                    "doc": doc,
                    "dense_rank": rank_0_idx + 1,
                    "sparse_rank": None
                }
            else:
                fused_docs[key]["dense_rank"] = rank_0_idx + 1
                
        # Sắp xếp Sparse Candidates toàn cục theo FTS score
        sparse_candidates.sort(key=lambda x: x.get("_score", x.get("score", 0.0)), reverse=True)
        for rank_0_idx, doc in enumerate(sparse_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {
                    "doc": doc,
                    "dense_rank": None,
                    "sparse_rank": rank_0_idx + 1
                }
            else:
                fused_docs[key]["sparse_rank"] = rank_0_idx + 1
                
        # Áp dụng công thức RRF và kết hợp prior weight
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
                "search_space_docs": search_space_docs
            }
            all_results.append(clean_doc)
            
        all_results.sort(key=lambda x: x['sw_rrf_score'], reverse=True)
        final_top_k = all_results[:top_k]
        
        return final_top_k, source_probs, emb[0]
