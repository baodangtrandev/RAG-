import os
import sys

# Thêm thư mục gốc vào sys.path để fix lỗi ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lancedb
import numpy as np
import logging
from typing import List, Dict, Any
from src.models.router_inference import ProbabilisticSourceRouter

logger = logging.getLogger(__name__)

class EnterpriseRetriever:
    """
    Retriever Pipeline chuẩn Production áp dụng kiến trúc T-RAG.
    Bao gồm:
    1. Probabilistic Source Router (PSR): Cắt giảm không gian tìm kiếm (Search Space).
    2. Source-Weighted Reciprocal Rank Fusion (SW-RRF): Xếp hạng tài liệu bằng thuật toán RRF kết hợp Bayesian Prior (Xác suất của nguồn).
    """
    def __init__(self, db_uri: str = "data/lancedb", model_dir: str = "models/psr_v2", tau: float = 0.15, gamma: float = 2.0, k_rrf: int = 60):
        self.db_uri = db_uri
        self.tau = tau
        self.gamma = gamma # Hệ số khuếch đại (Source Bias Factor)
        self.k_rrf = k_rrf
        
        logger.info(f"Khởi tạo Enterprise Retriever. Kết nối DB: {db_uri}")
        self.db = lancedb.connect(db_uri)
        
        logger.info("Khởi tạo PSR Router...")
        self.router = ProbabilisticSourceRouter(model_dir=model_dir)
        
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
            best_source = max(source_probs, key=source_probs.get)
            active_shards.append(best_source)
            
        logger.info(f"🔍 [Query]: '{query}'")
        logger.info(f"🛣️ [Router]: Quét {len(active_shards)}/9 bảng -> {active_shards}")
        
        all_results = []
        
        # Bước 3: Tìm kiếm Vector cục bộ & Áp dụng Source-Weighted RRF (SW-RRF)
        for source in active_shards:
            if source not in self.db.table_names():
                logger.error(f"LỖI: Không tìm thấy bảng '{source}' trong LanceDB.")
                continue
                
            table = self.db.open_table(source)
            # Quét rộng hơn top_k một chút ở mỗi bảng để đảm bảo RRF công bằng
            search_limit = max(top_k * 2, 10) 
            try:
                # Vector Search (Dense)
                results = table.search(emb[0]).limit(search_limit).to_list()
            except Exception as e:
                logger.error(f"Lỗi khi search bảng {source}: {e}")
                continue
            
            p_s = source_probs[source]
            # Tính Bayesian Prior weight (Xác suất nguồn ^ Gamma)
            prior_weight = p_s ** self.gamma
            
            for rank_0_idx, doc in enumerate(results):
                r_dense = rank_0_idx + 1 # Rank bắt đầu từ 1
                
                # Thuật toán SW-RRF cốt lõi:
                # Score = P(Source|Query)^Gamma * (1 / (k + Rank))
                rrf_score = 1.0 / (self.k_rrf + r_dense)
                sw_rrf_score = prior_weight * rrf_score
                
                # Đóng gói dữ liệu
                clean_doc = {
                    "source": source,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "text": doc.get("text", doc.get("content", "")),
                    "title": doc.get("title", ""),
                    "vector_distance": doc.get("_distance", 1.0), # L2 distance
                    "router_prob": p_s,
                    "original_rank": r_dense,
                    "sw_rrf_score": sw_rrf_score
                }
                all_results.append(clean_doc)
                
        # Bước 4: Xếp hạng Toàn cầu (Global Reranking)
        # Sắp xếp các tài liệu từ tất cả các Shard dựa trên điểm SW-RRF tổng hợp
        all_results.sort(key=lambda x: x['sw_rrf_score'], reverse=True)
        
        final_top_k = all_results[:top_k]
        
        logger.info(f"✨ [Retrieved]: Trả về {len(final_top_k)} tài liệu có độ tin cậy cao nhất.")
        return final_top_k

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # Giả định LanceDB của bạn nằm ở /network-volume/RAG-/data/lancedb
    db_path = "/network-volume/RAG-/data/lancedb"
    retriever = EnterpriseRetriever(db_uri=db_path, tau=0.15, gamma=2.0)
    
    test_queries = [
        "How do I fix the CI/CD pipeline out of memory error?",
        "What are the pricing tiers for HubSpot?",
        "Can someone send me the meeting notes from yesterday?"
    ]
    output_str = "\n" + "="*80 + "\n"
    output_str += "🚀 KẾT QUẢ TÌM KIẾM PIPELINE (SW-RRF)\n"
    output_str += "="*80 + "\n"
    
    for q in test_queries:
        docs = retriever.retrieve(q, top_k=3)
        output_str += f"\n❓ Câu hỏi: {q}\n"
        for i, doc in enumerate(docs):
            output_str += f"  [{i+1}] Nguồn: {doc['source'].upper()} (Độ tin cậy Router: {doc['router_prob']:.2f}) | Điểm SW-RRF: {doc['sw_rrf_score']:.6f}\n"
            # Cắt bớt các ký tự xuống dòng để in ra file đẹp hơn
            clean_text = doc['text'].replace('\n', ' ')
            output_str += f"      Text snippet: {clean_text}...\n"
            
    output_str += "="*80 + "\n"
    
    print(output_str)
    
    out_file = "models/psr_v2/retriever_demo_results.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_str)
        
    print(f"✅ Đã lưu toàn bộ kết quả Retrieval demo vào file: {out_file}")
