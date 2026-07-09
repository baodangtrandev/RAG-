import os
import argparse
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import torch

# Thêm TRAG-v2 vào sys.path để import từ src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.lance_schema import DocumentSchema

def main(data_dir, file_name, db_path, batch_size, model_name):
    # Ưu tiên chạy trên GPU (H100 của bạn sẽ gánh mượt mà)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading embedding model: {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"🔗 Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)
    
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
        
    print(f"📂 Reading chunked data from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Xử lý vấn đề array object từ Pandas parquet (nếu có)
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    source_types = df['source_type'].unique()
    print(f"✅ Found {len(source_types)} unique source types for Sharding: {source_types}")
    
    # SHARDING THEO NGUỒN VẬT LÝ
    for src in source_types:
        table_name = str(src).lower().strip().replace(" ", "_").replace("-", "_")
        
        subset = df[df['source_type'] == src]
        total_docs = len(subset)
        print(f"\n--- ⚡ Processing source (Table): {table_name} | {total_docs} chunks ---")
        
        tbl = None
        if table_name in db.table_names():
            tbl = db.open_table(table_name)
            print(f"ℹ️ Table '{table_name}' already exists in LanceDB. Appending data...")
            
        for i in tqdm(range(0, total_docs, batch_size), desc=f"Ingesting {table_name}"):
            batch = subset.iloc[i:i+batch_size].copy()
            
            # Embed content bằng BGE-Large
            texts = batch['content'].fillna("").astype(str).tolist()
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
            
            # Chuẩn bị Data Payload
            data = []
            for j, (_, row) in enumerate(batch.iterrows()):
                data.append({
                    "doc_id": str(row['doc_id']),
                    "original_doc_id": str(row.get('original_doc_id', row['doc_id'])),
                    "chunk_id": int(row.get('chunk_id', 0)),
                    "content": str(row['content']),
                    "title": str(row['title']) if pd.notna(row['title']) else None,
                    "source_type": str(row['source_type']),
                    "vector": embeddings[j].tolist()
                })
            
            # Ghi vào DB
            if tbl is None:
                # Tạo bảng mới nếu chưa có
                tbl = db.create_table(table_name, schema=DocumentSchema, data=data)
            else:
                # Thêm vào bảng đã có
                tbl.add(data)

    print("\n🎉 HOÀN THÀNH INGESTION TO LANCEDB (SHARDED)!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/EnterpriseRAG-Bench/data/documents", help="Path to documents dir")
    parser.add_argument("--file-name", type=str, default="test_chunked_v2.parquet", help="Parquet file name")
    parser.add_argument("--db-path", type=str, default="./data/lancedb", help="Path to lancedb")
    # Tăng batch_size lên 1024 vì bạn xài H100 GPU cực mạnh
    parser.add_argument("--batch-size", type=int, default=1024)
    # Model BGE được mệnh danh là một trong những model xịn nhất cho Retrieval
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    
    args = parser.parse_args()
    main(args.data_dir, args.file_name, args.db_path, args.batch_size, args.embedding_model)
