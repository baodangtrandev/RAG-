import argparse
import json
import logging
import os

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

USE_CUML = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    queries = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q = data.get("question", "")
            sources = data.get("source_types", [])
            if q and sources:
                queries.append(q)
                labels.append([s.lower().strip() for s in sources])

    return queries, labels


def main(data_path: str, model_save_dir: str, embedding_model_name: str):
    logger.info("=== STEP 1: LOAD DATA & FEATURE EXTRACTION ===")
    queries, labels = load_jsonl_dataset(data_path)
    logger.info(f"Loaded {len(queries)} samples from {data_path}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Encoder Model '{embedding_model_name}' on {device} (FP16)...")

    # Sử dụng torch.float16 để tối ưu tốc độ/VRAM
    encoder = SentenceTransformer(embedding_model_name, device=device, model_kwargs={"torch_dtype": torch.float16})

    logger.info("Đang chuyển đổi Câu hỏi thành Vectors (Pre-computing Embeddings)...")
    # Batch size lớn vì dữ liệu là câu hỏi ngắn, FP16 trên H100 chạy cực nhanh
    X_embeddings = encoder.encode(queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info("Chuyển đổi nhãn sang Multi-Label Binarizer (Sigmoid)...")
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(labels)

    logger.info("=== STEP 2: DATA SPLITTING & MODEL TRAINING ===")
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_encoded, test_size=0.2, random_state=42)

    logger.info(f"Train size: {X_train.shape[0]} samples. Test size: {X_test.shape[0]} samples.")

    base_clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
    clf = OneVsRestClassifier(base_clf)

    logger.info("Training OneVsRestClassifier...")
    clf.fit(X_train, y_train)

    logger.info("=== STEP 3: MLOps EVALUATION ===")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    classes = mlb.classes_

    report_str = f"Exact Match Accuracy: {acc:.4f}\n"
    report_str += f"Log-Loss: {loss:.4f}\n\n"
    report_str += "--- Multi-Label Classification Report ---\n"
    report_str += classification_report(y_test, y_pred, target_names=classes) + "\n\n"

    report_str += "\n=== RAG ROUTER EVALUATION ===\n"

    sorted_probs = np.argsort(y_pred_proba, axis=1)[:, ::-1]

    report_str += "--- Routing Performance (Tau Thresholds) ---\n"
    taus = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for tau in taus:
        active_shards_per_query = np.sum(y_pred_proba >= tau, axis=1)
        active_shards_per_query = np.maximum(active_shards_per_query, 1)
        avg_shards = np.mean(active_shards_per_query)

        hits = 0
        for i in range(len(y_test)):
            true_indices = np.where(y_test[i] == 1)[0]
            if len(true_indices) == 0:
                continue

            if np.any(y_pred_proba[i, true_indices] >= tau):
                hits += 1
            elif sorted_probs[i, 0] in true_indices:
                hits += 1

        hit_rate = hits / len(y_test)

        report_str += f"Tau = {tau:.2f} | Hit Rate: {hit_rate*100:.1f}% | Avg Shards: {avg_shards:.2f}/9\n"

    os.makedirs(model_save_dir, exist_ok=True)
    report_path = os.path.join(model_save_dir, "evaluation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    logger.info(f"Evaluation report saved to: {report_path}")

    logger.info("=== STEP 4: ARTIFACT MANAGEMENT ===")
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, "psr_router.joblib")
    classes_path = os.path.join(model_save_dir, "psr_classes.json")

    joblib.dump(clf, model_path)

    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(list(classes), f)

    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Classes: {list(classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Probabilistic Source Router (PSR)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/router_training_data_v1.jsonl",
        help="Đường dẫn file JSONL chứa training data",
    )
    parser.add_argument("--model-save-dir", type=str, default="models/psr_v2", help="Thư mục lưu mô hình")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Mô hình Embedding để trích xuất đặc trưng",
    )

    args = parser.parse_args()
    main(args.data_path, args.model_save_dir, args.embedding_model)
