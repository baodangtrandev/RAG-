import os
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# Settings
DATA_DIR = "../all_documents/all_documents"  #path của folder `all_documents` đã được extract
DB_PATH = "../lancedb_data"
TABLE_NAME = "enterprise_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 128

def get_db_and_table():
    db = lancedb.connect(DB_PATH)
    return db

def ingest_data():
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    # By default, sentence-transformers uses GPU if available
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    db = get_db_and_table()
    
    # Define PyArrow schema for the table
    schema = pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("title", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    if TABLE_NAME in db.table_names():
        print(f"Table '{TABLE_NAME}' already exists. Overwriting...")
        db.drop_table(TABLE_NAME)
        
    table = db.create_table(TABLE_NAME, schema=schema)
    
    print(f"Scanning directory {DATA_DIR}...")
    # Thu thập tất cả các file .txt
    txt_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
                
    print(f"Found {len(txt_files)} text files in directory.")
    
    batch_docs = []
    batch_texts_for_embed = []
    count = 0
    
    for file_path in txt_files:
        # Lấy source_type từ tên thư mục con ngay dưới DATA_DIR
        rel_path = os.path.relpath(file_path, DATA_DIR)
        parts = rel_path.split(os.sep)
        source_type = parts[0]
        filename = parts[-1]
        
        # Extract doc_id
        if "__" in filename:
            doc_id = filename.split('__')[0]
        else:
            doc_id = filename.split('_')[0] if filename.startswith("dsid_") else filename
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        lines = text.split('\n')
        title = lines[0].strip() if lines else ""
        
        # Để chạy Embedding, ta nối title và nội dung
        embed_text = title + "\n" + text
        
        doc_entry = {
            "doc_id": doc_id,
            "source_type": source_type,
            "title": title,
            "content": text
        }
        batch_docs.append(doc_entry)
        batch_texts_for_embed.append(embed_text)
        count += 1
        
        if len(batch_docs) >= BATCH_SIZE:
            # Chạy model encode
            embeddings = model.encode(batch_texts_for_embed, normalize_embeddings=True)
            
            # Gắn vector vào data
            for i, doc in enumerate(batch_docs):
                doc["vector"] = embeddings[i].tolist()
            
            # Chèn vào LanceDB
            table.add(batch_docs)
            print(f"Ingested {count} / {len(txt_files)} documents...")
            
            # Reset batch
            batch_docs = []
            batch_texts_for_embed = []
            
    # Ingest nốt phần dư
    if batch_docs:
        embeddings = model.encode(batch_texts_for_embed, normalize_embeddings=True)
        for i, doc in enumerate(batch_docs):
            doc["vector"] = embeddings[i].tolist()
        table.add(batch_docs)
        print(f"Ingested {count} / {len(txt_files)} documents...")

    print("Building Full-Text Search (FTS) index via tantivy...")
    table.create_fts_index("content", replace=True)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
