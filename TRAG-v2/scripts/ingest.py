import os
import argparse
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys


# python scripts/ingest.py \
#   --data-dir data/EnterpriseRAG-Bench/data/documents \
#   --db-path ./data/lancedb \
#   --batch-size 512 \
#   --embedding-model BAAI/bge-large-en-v1.5


# Thêm TRAG-v2 vào sys.path để import từ src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.lance_schema import DocumentSchema

def main(data_dir, db_path, batch_size, model_name):
    # print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # print(f"Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)
    
    file_path = os.path.join(data_dir, "test.parquet")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    # print(f"Reading data from {file_path}")
    df = pd.read_parquet(file_path)
    
    # Xử lý vấn đề array object từ EDA
    # Vì file parquet có thể lưu numpy array cho string
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    source_types = df['source_type'].unique()
    print(f"Found {len(source_types)} unique source types: {source_types}")
    
    # Sharding theo nguồn
    for src in source_types:
        print(f"\n--- Processing source: {src} ---")
        table_name = str(src).lower().replace(" ", "_").replace("-", "_")
        
        subset = df[df['source_type'] == src]
        total_docs = len(subset)
        print(f"Total documents for {src}: {total_docs}")
        
        tbl = None
        if table_name in db.table_names():
            tbl = db.open_table(table_name)
            print(f"Table '{table_name}' already exists. Will append.")
            
        for i in tqdm(range(0, total_docs, batch_size), desc=f"Ingesting {table_name}"):
            batch = subset.iloc[i:i+batch_size].copy()
            
            # Embed content (BGE-M3 or BGE-large)
            texts = batch['content'].fillna("").astype(str).tolist()
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # Prepare data
            data = []
            for j, (_, row) in enumerate(batch.iterrows()):
                data.append({
                    "doc_id": str(row['doc_id']),
                    "content": str(row['content']),
                    "title": str(row['title']) if pd.notna(row['title']) else None,
                    "source_type": str(row['source_type']),
                    "vector": embeddings[j].tolist()
                })
            
            if tbl is None:
                tbl = db.create_table(table_name, schema=DocumentSchema, data=data)
            else:
                tbl.add(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to documents dir")
    parser.add_argument("--db-path", type=str, required=True, help="Path to lancedb")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    
    args = parser.parse_args()
    main(args.data_dir, args.db_path, args.batch_size, args.embedding_model)
