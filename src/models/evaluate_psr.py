import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import argparse

def evaluate(model_dir, test_file, embedding_model):
    print(f"Đang tải tập TEST từ {test_file}...")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Không tìm thấy file: {test_file}")
        
    df = pd.read_parquet(test_file)
    
    queries = df['question'].tolist()
    
    true_sources = []
    for s in df['source_types']:
        if isinstance(s, (list, np.ndarray)):
            true_sources.append([str(x).lower().strip() for x in s])
        else:
            true_sources.append([str(s).lower().strip()])
            
    print(f"✅ Đã tải {len(queries)} câu hỏi Test khó.")
    
    print(f"Đang tải Embedding model {embedding_model} lên GPU (FP16)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(embedding_model, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("Đang biến đổi Câu hỏi Test thành Vector...")
    X_test = encoder.encode(queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    model_path = os.path.join(model_dir, "psr_router.joblib")
    classes_path = os.path.join(model_dir, "psr_classes.json")
    
    print(f"Đang nạp mô hình RAG Router V2 từ {model_path}...")
    clf = joblib.load(model_path)
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
        
    print("Đang dự đoán Xác suất phân phối (Routing)...")
    y_pred_proba = clf.predict_proba(X_test)
    
    # Lấy index sắp xếp giảm dần (để lấy Top-1)
    sorted_probs = np.argsort(y_pred_proba, axis=1)[:, ::-1]
    
    report_str = "="*60 + "\n"
    report_str += "🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST THỰC TẾ (GOLD STANDARD)\n"
    report_str += "="*60 + "\n\n"
    
    print(report_str, end="")
    
    # Đánh giá trên các ngưỡng Tau
    taus = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for tau in taus:
        # Số lượng shard dự kiến gọi
        active_shards_per_query = np.sum(y_pred_proba >= tau, axis=1)
        active_shards_per_query = np.maximum(active_shards_per_query, 1) # Fallback: ít nhất lấy Top-1
        avg_shards = np.mean(active_shards_per_query)
        
        # Đếm số lượng Hit
        hits = 0
        valid_queries = 0
        for i in range(len(queries)):
            expected = true_sources[i]
            # Chuyển đổi tên source sang index
            expected_indices = [classes.index(src) for src in expected if src in classes]
            
            if not expected_indices:
                continue
                
            valid_queries += 1
            
            # Hit Rate: Có bất kỳ đáp án đúng nào có xác suất >= Tau không?
            if np.any(y_pred_proba[i, expected_indices] >= tau):
                hits += 1
            # Check Fallback: Hoặc nếu đáp án đúng lại chính là Top-1
            elif sorted_probs[i, 0] in expected_indices:
                hits += 1
                
        if valid_queries > 0:
            hit_rate = hits / valid_queries
            line = f"Ngưỡng Tau = {tau:.2f} | Hit Rate (Tỷ lệ trúng): {hit_rate*100:.1f}% | Số Shards truy vấn trung bình: {avg_shards:.2f}/9\n"
            report_str += line
            print(line, end="")

    out_file = os.path.join(model_dir, "test_evaluation_report.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n✅ Đã lưu toàn bộ kết quả vào file: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/psr_v2")
    parser.add_argument("--test-file", default="data/EnterpriseRAG-Bench/data/questions/test.parquet")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()
    evaluate(args.model_dir, args.test_file, args.embedding_model)
