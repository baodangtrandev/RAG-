import os
import sys
import json
import argparse
import logging
import joblib
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
# Tạm thời vô hiệu hóa cuML do lỗi CUDA_ERROR_INVALID_IMAGE (Xung đột Driver/CuPy trên card H100)
# scikit-learn (CPU) vẫn sẽ chạy cực nhanh với lượng data này (dưới 1 giây).
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
USE_CUML = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix

# Cấu hình Logging chuẩn Enterprise
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_jsonl_dataset(file_path: str):
    """
    Đọc dữ liệu từ file JSONL, trích xuất Query và Target Source.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {file_path}")
        
    queries = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Lấy câu hỏi
            q = data.get('question', '')
            # Lấy nhãn (Nguồn đầu tiên trong danh sách source_types)
            sources = data.get('source_types', [])
            if q and sources:
                queries.append(q)
                # Lưu dưới dạng list các sources để phục vụ Multi-Label
                labels.append([s.lower().strip() for s in sources])
                
    return queries, labels

def main(data_path: str, model_save_dir: str, embedding_model_name: str):
    logger.info("=== BƯỚC 1: LOAD DỮ LIỆU & FEATURE EXTRACTION ===")
    queries, labels = load_jsonl_dataset(data_path)
    logger.info(f"Đã load {len(queries)} samples từ {data_path}.")
    
    # Khởi tạo mô hình Embedding (Phải dùng đúng mô hình BGE-Large-en-v1.5 đã dùng lúc Ingest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Đang tải Encoder Model '{embedding_model_name}' lên {device} (FP16)...")
    
    # Sử dụng torch.float16 để tối ưu tốc độ/VRAM
    encoder = SentenceTransformer(embedding_model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    logger.info("Đang chuyển đổi Câu hỏi thành Vectors (Pre-computing Embeddings)...")
    # Batch size lớn vì dữ liệu là câu hỏi ngắn, FP16 trên H100 chạy cực nhanh
    X_embeddings = encoder.encode(queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info("Chuyển đổi nhãn sang Multi-Label Binarizer (Sigmoid)...")
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(labels)
    
    logger.info("=== BƯỚC 2: DATA SPLITTING & MODEL TRAINING ===")
    # Multi-label không dùng stratify đơn giản được, chia ngẫu nhiên
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_encoded, test_size=0.2, random_state=42)
    
    logger.info(f"Kích thước tập Train: {X_train.shape[0]} samples. Tập Test: {X_test.shape[0]} samples.")
    
    # Khởi tạo Logistic Regression với OneVsRestClassifier (Sigmoid)
    logger.info("Chạy Multi-Label OneVsRestClassifier (Sigmoid) thay vì Softmax...")
    base_clf = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        C=0.5, 
        class_weight='balanced',
        random_state=42
    )
    clf = OneVsRestClassifier(base_clf)
        
    logger.info("Đang huấn luyện mô hình Toán học phân phối Xác suất độc lập (Sigmoid)...")
    clf.fit(X_train, y_train)
    
    logger.info("=== BƯỚC 3: MLOps EVALUATION (KIỂM ĐỊNH CHẤT LƯỢNG) ===")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Exact match accuracy
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    classes = mlb.classes_
    
    # Gom toàn bộ báo cáo thành chuỗi
    report_str = f"Độ chính xác khớp hoàn toàn (Exact Match Accuracy): {acc:.4f}\n"
    report_str += f"Log-Loss: {loss:.4f}\n\n"
    
    report_str += "--- Multi-Label Classification Report ---\n"
    report_str += classification_report(y_test, y_pred, target_names=classes) + "\n\n"
    
    # --- ĐÁNH GIÁ ĐẶC THÙ CHO RAG ROUTER ---
    report_str += "\n=== ĐÁNH GIÁ CHUYÊN SÂU CHO HỆ THỐNG RAG ROUTER (MULTI-LABEL) ===\n"
    
    # Tính Top-1 và Top-2 Accuracy
    sorted_probs = np.argsort(y_pred_proba, axis=1)[:, ::-1] 
    
    # Đối với Multi-label, Hit Rate được tính nếu BẤT KỲ nhãn đúng nào lọt vào tập kết quả
    report_str += "--- Phân tích Hiệu suất Routing (Theo Threshold Tau) ---\n"
    taus = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for tau in taus:
        active_shards_per_query = np.sum(y_pred_proba >= tau, axis=1)
        active_shards_per_query = np.maximum(active_shards_per_query, 1) # Fallback to Top-1
        avg_shards = np.mean(active_shards_per_query)
        
        # Hit Rate: Có bao nhiêu queries có ít nhất 1 true label vượt qua tau hoặc là top 1?
        # y_test là ma trận binary (N, C)
        hits = 0
        for i in range(len(y_test)):
            true_indices = np.where(y_test[i] == 1)[0]
            if len(true_indices) == 0:
                continue
            
            # Check nếu bất kỳ true label nào có xác suất >= tau
            if np.any(y_pred_proba[i, true_indices] >= tau):
                hits += 1
            # Check fallback top 1
            elif sorted_probs[i, 0] in true_indices:
                hits += 1
                
        hit_rate = hits / len(y_test)
        
        report_str += f"Ngưỡng Tau = {tau:.2f} | Hit Rate (Tỷ lệ trúng đích): {hit_rate*100:.1f}% | Số Shards truy vấn trung bình: {avg_shards:.2f}/9\n"
        
    # Tạo thư mục trước nếu chưa có
    os.makedirs(model_save_dir, exist_ok=True)
    report_path = os.path.join(model_save_dir, "evaluation_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    logger.info(f"Đã lưu kết quả Đánh giá chi tiết (Evaluation Report) tại: {report_path}")
    
    logger.info("=== BƯỚC 4: ĐÓNG GÓI & LƯU TRỮ MÔ HÌNH (ARTIFACT MANAGEMENT) ===")
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_path = os.path.join(model_save_dir, "psr_router.joblib")
    classes_path = os.path.join(model_save_dir, "psr_classes.json")
    
    # Lưu trọng số W, b
    joblib.dump(clf, model_path)
    
    # Lưu danh sách tên các Source (để mapping lại thứ tự class)
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(list(classes), f)
        
    logger.info(f"Đã lưu thành công Mô hình tại: {model_path}")
    logger.info(f"Danh sách Classes: {list(classes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Probabilistic Source Router (PSR)")
    parser.add_argument("--data-path", type=str, default="data/router_training_data_v1.jsonl", help="Đường dẫn file JSONL chứa training data")
    parser.add_argument("--model-save-dir", type=str, default="models/psr_v2", help="Thư mục lưu mô hình")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5", help="Mô hình Embedding để trích xuất đặc trưng")
    
    args = parser.parse_args()
    main(args.data_path, args.model_save_dir, args.embedding_model)
