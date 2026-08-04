import json
import logging
import os

import joblib
import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class ProbabilisticSourceRouter:
    """Probabilistic Source Router for multi-label source prediction."""

    def __init__(self, model_dir: str = None, embedding_model: str = None):
        self.model_dir = model_dir or os.environ.get("PSR_MODEL_DIR", "models/psr_v2")
        embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

        classes_path = os.path.join(self.model_dir, "psr_classes.json")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes config not found: {classes_path}")
        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        model_path = os.path.join(self.model_dir, "psr_router.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.clf = joblib.load(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.encoder = SentenceTransformer(
            embedding_model, device=self.device, model_kwargs={"torch_dtype": torch.float16}
        )

    def route(self, query: str, tau: float = 0.15) -> list[str]:
        emb = self.encoder.encode([query], normalize_embeddings=True)
        probs = self.clf.predict_proba(emb)[0]

        # 3. Lọc các nguồn vượt qua ngưỡng Threshold Tau
        active_shards = []
        for idx, prob in enumerate(probs):
            if prob >= tau:
                active_shards.append(self.classes[idx])

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
        "Who is assigned to the API bug ticket?",
    ]

    print("\n--- BẮT ĐẦU ROUTING TEST ---")
    for q in test_queries:
        shards = router.route(q, tau=0.15)
        print(f"\nQuery: {q}")
        print(f"👉 Kích hoạt các bảng: {shards}")
