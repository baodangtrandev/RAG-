import os
import json
import joblib
import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ProbabilisticSourceRouter:
    """
    RAG Router dựa trên xác suất (Probabilistic Source Router).
    Sử dụng Multi-Label Sigmoid Classifier để điều phối câu hỏi vào đúng các Shard dữ liệu.
    """
    def __init__(self, model_dir: str = "models/psr_v2", embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.model_dir = model_dir
        
        # 1. Load danh sách tên các nguồn (Classes)
        classes_path = os.path.join(model_dir, "psr_classes.json")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Không tìm thấy cấu hình lớp: {classes_path}")
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)
            
        # 2. Load trọng số mô hình Logistic Regression (Sigmoid)
        model_path = os.path.join(model_dir, "psr_router.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
        self.clf = joblib.load(model_path)
        
        # 3. Khởi tạo mô hình Embedding (chỉ để Inference)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Bật cấu hình an toàn cho multiprocessing nếu sau này chạy server
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.encoder = SentenceTransformer(embedding_model, device=self.device, model_kwargs={"torch_dtype": torch.float16})
        
    def route(self, query: str, tau: float = 0.15) -> list[str]:
        """
        Nhận vào câu hỏi (Query) và trả về danh sách các Shard (Tables) cần tìm kiếm.
        :param query: Câu hỏi của người dùng.
        :param tau: Ngưỡng xác suất để kích hoạt một nguồn (0.0 -> 1.0).
        :return: Danh sách tên các table cần quét.
        """
        # 1. Encode query thành vector
        emb = self.encoder.encode([query], normalize_embeddings=True)
        
        # 2. Lấy xác suất phân phối qua hàm Sigmoid
        probs = self.clf.predict_proba(emb)[0]
        
        # 3. Lọc các nguồn vượt qua ngưỡng Threshold Tau
        active_shards = []
        for idx, prob in enumerate(probs):
            if prob >= tau:
                active_shards.append(self.classes[idx])
                
        # 4. Cơ chế an toàn (Fallback): Nếu câu hỏi quá mờ nhạt, lấy mặc định nguồn có xác suất cao nhất (Top-1)
        if not active_shards:
            best_idx = np.argmax(probs)
            active_shards.append(self.classes[best_idx])
            
        return active_shards

if __name__ == "__main__":
    # Test thử trực tiếp
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    print("Đang khởi động hệ thống Probabilistic Source Router...")
    router = ProbabilisticSourceRouter(model_dir="models/psr_v2")
    
    test_queries = [
        "How do I fix the Out of Memory error in our CI/CD Github Actions?",
        "What are the pricing tiers for our HubSpot CRM integration?",
        "Can someone send me the meeting notes from yesterday's all-hands?",
        "Who is assigned to the API bug ticket?"
    ]
    
    print("\n--- BẮT ĐẦU ROUTING TEST ---")
    for q in test_queries:
        shards = router.route(q, tau=0.15)
        print(f"\nQuery: {q}")
        print(f"👉 Kích hoạt các bảng: {shards}")
