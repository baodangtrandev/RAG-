import os
import argparse
import pandas as pd
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from tqdm import tqdm
import logging


# python scripts/semantic_chunker.py \
#   --input-file data/EnterpriseRAG-Bench/data/documents/test.parquet \
#   --output-file data/EnterpriseRAG-Bench/data/documents/test_chunked.parquet \
#   --embedding-model BAAI/bge-large-en-v1.5


def main(input_file, output_file, model_name):
    print(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model {model_name} on {device} for Semantic Chunking")

    embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
    
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=embed_model
    )
    
    new_rows = []
    
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = str(row['content']) if pd.notna(row['content']) else ""
        title = str(row['title']) if 'title' in row and pd.notna(row['title']) else ""
        
        if not content.strip():
            continue
            
        # Nối title vào đầu nội dung (nếu có title) để làm giàu ngữ cảnh cho từng chunk
        if title.strip():
            full_content = f"Title: {title}\n\n{content}"
        else:
            full_content = content
            
        doc = Document(text=full_content)
        
        nodes = splitter.get_nodes_from_documents([doc])
        

        for i, node in enumerate(nodes):
            new_row = row.copy()
            new_row['content'] = node.get_content()
            new_row['chunk_id'] = i
            # Tạo doc_id mới duy nhất nhưng vẫn giữ lại original_doc_id
            new_row['original_doc_id'] = row['doc_id']
            new_row['doc_id'] = f"{row['doc_id']}_chunk{i}"
            
            new_rows.append(new_row)
            
    chunked_df = pd.DataFrame(new_rows)
    print(f"\nOriginal documents: {len(df)}")
    print(f"Chunked documents: {len(chunked_df)}")
    
    print(f"Saving chunked data to {output_file}")
    chunked_df.to_parquet(output_file)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Đường dẫn đến file parquet gốc (chưa chunk)")
    parser.add_argument("--output-file", type=str, required=True, help="Đường dẫn xuất file parquet sau khi chunk")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.embedding_model)
