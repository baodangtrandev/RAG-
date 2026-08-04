import os
import sys

# Thêm thư mục gốc vào sys.path để fix lỗi ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Any, Dict, List, Optional

import lancedb

from src.models.router_inference import ProbabilisticSourceRouter

logger = logging.getLogger(__name__)


class EnterpriseRetriever:
    """
    Retriever Pipeline chuẩn Production áp dụng kiến trúc T-RAG.
    Bao gồm:
    1. Probabilistic Source Router (PSR): Cắt giảm không gian tìm kiếm (Search Space).
    2. Source-Weighted Reciprocal Rank Fusion (SW-RRF): Xếp hạng tài liệu bằng thuật toán RRF kết hợp Bayesian Prior (Xác suất của nguồn).
    """

    def __init__(
        self,
        db_uri: Optional[str] = None,
        model_dir: Optional[str] = None,
        tau: Optional[float] = None,
        gamma: Optional[float] = None,
        k_rrf: Optional[int] = None,
    ):
        from dotenv import load_dotenv

        load_dotenv()

        self.db_uri = db_uri or os.environ.get("RAG_DB_URI", "data/lancedb")
        model_dir = model_dir or os.environ.get("PSR_MODEL_DIR", "models/psr_v2")

        try:
            self.tau = tau if tau is not None else float(os.environ.get("RAG_TAU", "0.15"))
        except ValueError:
            self.tau = 0.15

        try:
            self.gamma = gamma if gamma is not None else float(os.environ.get("RAG_GAMMA", "2.0"))
        except ValueError:
            self.gamma = 2.0  # Hệ số khuếch đại (Source Bias Factor)

        try:
            self.k_rrf = k_rrf if k_rrf is not None else int(os.environ.get("RAG_K_RRF", "60"))
        except ValueError:
            self.k_rrf = 60

        logger.info(f"Khởi tạo Enterprise Retriever. Kết nối DB: {self.db_uri}")
        self.db = lancedb.connect(self.db_uri)

        logger.info("Khởi tạo PSR Router...")
        self.router = ProbabilisticSourceRouter(model_dir=model_dir)

        # Cache tables to avoid disk I/O bottleneck
        self.table_names = self.db.table_names()
        self.tables = {name: self.db.open_table(name) for name in self.table_names}

        try:
            self.hybrid_search = os.environ.get("RAG_HYBRID_SEARCH", "True").lower() == "true"
        except Exception:
            self.hybrid_search = True
        logger.info(f"Cấu hình Hybrid Search: {self.hybrid_search}")

        try:
            self.dense_weight = float(os.environ.get("RAG_DENSE_WEIGHT", "0.3"))
        except ValueError:
            self.dense_weight = 0.3

        try:
            self.sparse_weight = float(os.environ.get("RAG_SPARSE_WEIGHT", "0.7"))
        except ValueError:
            self.sparse_weight = 0.7
        logger.info(f"Trọng số Hybrid Search - Dense: {self.dense_weight}, Sparse: {self.sparse_weight}")

        # Get document counts for search space calculation
        self.table_sizes = {}
        for name, table in self.tables.items():
            try:
                self.table_sizes[name] = len(table)
            except Exception:
                self.table_sizes[name] = 0

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tài liệu qua 3 bước Toán học.
        """
        # Bước 1: Mã hóa Vector và tính Xác suất qua Router
        emb = self.router.encoder.encode([query], normalize_embeddings=True)
        probs = self.router.clf.predict_proba(emb)[0]

        # Mapping Probability với Từng Bảng
        source_probs = {self.router.classes[i]: float(probs[i]) for i in range(len(probs))}

        # Bước 2: Kích hoạt Shards (Sub-space Search)
        active_shards = [s for s, p in source_probs.items() if p >= self.tau]

        if not active_shards:
            logger.warning("Không có Shard nào vượt qua Threshold. Kích hoạt Fallback (Top-1).")
            best_source = max(source_probs, key=source_probs.__getitem__)
            active_shards.append(best_source)

        logger.info(f"🔍 [Query]: '{query}'")
        logger.info(f"🛣️ [Router]: Quét {len(active_shards)}/9 bảng -> {active_shards}")

        search_space_docs = sum(self.table_sizes.get(source, 0) for source in active_shards)

        search_limit = max(top_k * 2, 10)
        dense_candidates = []
        sparse_candidates = []

        # Bước 3: Tìm kiếm song song Dense & Sparse cục bộ
        for source in active_shards:
            if source not in self.tables:
                logger.error(f"LỖI: Không tìm thấy bảng '{source}' trong LanceDB.")
                continue

            table = self.tables[source]
            p_s = source_probs[source]
            prior_weight = p_s**self.gamma

            # 3.1 Vector Search (Dense)
            try:
                results = table.search(emb[0]).limit(search_limit).to_list()
                for doc in results:
                    doc["_source"] = source
                    doc["_prior_weight"] = prior_weight
                    doc["_router_prob"] = p_s
                    dense_candidates.append(doc)
            except Exception as e:
                logger.error(f"Lỗi khi vector search bảng {source}: {e}")

            # 3.2 FTS Search (Sparse)
            if self.hybrid_search:
                try:
                    results_fts = table.search(query, query_type="fts").limit(search_limit).to_list()
                    for doc in results_fts:
                        doc["_source"] = source
                        doc["_prior_weight"] = prior_weight
                        doc["_router_prob"] = p_s
                        sparse_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Lỗi khi FTS search bảng {source}: {e}")

        # Bước 4: Xếp hạng Toàn cầu & Dung hợp bằng SW-RRF
        fused_docs = {}  # key: (source, doc_id) -> values: doc, dense_rank, sparse_rank

        # Sắp xếp Dense Candidates toàn cục theo distance (L2 distance càng nhỏ càng tốt/gần)
        dense_candidates.sort(key=lambda x: x.get("_distance", float("inf")))
        for rank_0_idx, doc in enumerate(dense_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": rank_0_idx + 1, "sparse_rank": None}
            else:
                fused_docs[key]["dense_rank"] = rank_0_idx + 1

        # Sắp xếp Sparse Candidates toàn cục theo FTS score (càng lớn càng tốt/khớp)
        sparse_candidates.sort(key=lambda x: x.get("_score", x.get("score", 0.0)), reverse=True)
        for rank_0_idx, doc in enumerate(sparse_candidates):
            source = doc["_source"]
            doc_id = doc.get("doc_id", "unknown")
            key = (source, doc_id)
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": None, "sparse_rank": rank_0_idx + 1}
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
                "search_space_docs": search_space_docs,
            }
            all_results.append(clean_doc)

        # Sắp xếp lại lần cuối theo điểm SW-RRF tổng hợp
        all_results.sort(key=lambda x: x["sw_rrf_score"], reverse=True)

        final_top_k = all_results[:top_k]

        logger.info(f"✨ [Retrieved]: Trả về {len(final_top_k)} tài liệu có độ tin cậy cao nhất.")
        return final_top_k


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Đọc cấu hình từ .env thay vì hardcode đường dẫn tuyệt đối
    retriever = EnterpriseRetriever()  # Tự động load RAG_DB_URI, RAG_TAU, RAG_GAMMA từ .env

    test_queries = [
        "How do I fix the CI/CD pipeline out of memory error?",
        "What are the pricing tiers for HubSpot?",
        "Can someone send me the meeting notes from yesterday?",
    ]
    output_str = "\n" + "=" * 80 + "\n"
    output_str += "🚀 KẾT QUẢ TÌM KIẾM PIPELINE (SW-RRF)\n"
    output_str += "=" * 80 + "\n"

    for q in test_queries:
        docs = retriever.retrieve(q, top_k=3)
        output_str += f"\n❓ Câu hỏi: {q}\n"
        for i, doc in enumerate(docs):
            output_str += f"  [{i+1}] Nguồn: {doc['source'].upper()} (Độ tin cậy Router: {doc['router_prob']:.2f}) | Điểm SW-RRF: {doc['sw_rrf_score']:.6f}\n"
            # Cắt bớt các ký tự xuống dòng để in ra file đẹp hơn
            clean_text = doc["content"].replace("\n", " ")
            output_str += f"      Text snippet: {clean_text[:200]}...\n"

    output_str += "=" * 80 + "\n"

    print(output_str)

    out_file = "models/psr_v2/retriever_demo_results.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_str)

    print(f"✅ Đã lưu toàn bộ kết quả Retrieval demo vào file: {out_file}")
