import os
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk
import time

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def safe_split_long_sentence(text, max_length=1500):
    """
    Cắt an toàn các câu quá dài bằng khoảng trắng, tránh cắt ngang từ (gây rác token).
    1500 ký tự ~ 300-400 tokens, đảm bảo an toàn cho max_length=512 của BGE.
    """
    if len(text) <= max_length:
        return [text]
    
    words = text.split(' ')
    chunks = []
    curr = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 > max_length:
            chunks.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr:
        chunks.append(" ".join(curr))
    return chunks

def split_into_sentences(text):
    """
    Cắt câu thông minh:
    1. Giữ nguyên dấu xuống dòng (Bảo toàn list, code block).
    2. Dùng NLTK cắt câu.
    3. Fallback xử lý câu quá dài một cách an toàn.
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    all_sentences = []
    for para in paragraphs:
        sentences = nltk.tokenize.sent_tokenize(para)
        for s in sentences:
            if len(s) > 1500:
                # Cắt an toàn theo khoảng trắng, tuyệt đối không dùng slice s[i:i+2000]
                sub_sentences = safe_split_long_sentence(s, max_length=1500)
                all_sentences.extend(sub_sentences)
            else:
                all_sentences.append(s)
                
    return [s.strip() for s in all_sentences if s.strip()]

def main(input_file, output_file, model_name, batch_size):
    print(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Fix array objects in pandas
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model {model_name} on {device}")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half() # Ép FP16 tăng x2 tốc độ, tiết kiệm VRAM
        
    new_rows = []
    
    # Gom 1000 docs mỗi mẻ để tiết kiệm RAM hệ thống
    doc_batch_size = 1000 
    
    for start_idx in tqdm(range(0, len(df), doc_batch_size), desc="Processing Batches"):
        t_start = time.time()
        batch_df = df.iloc[start_idx : start_idx+doc_batch_size]
        
        all_sentences = []
        doc_sentence_counts = []
        doc_metadata = []
        
        # 1. Split Sentences
        for idx, row in batch_df.iterrows():
            content = str(row.get('content', '')) if pd.notna(row.get('content')) else ""
            title = str(row.get('title', '')) if 'title' in row and pd.notna(row.get('title')) else ""
            
            if not content.strip():
                continue
                
            full_content = f"Title: {title}\n\n{content}" if title.strip() else content
            sentences = split_into_sentences(full_content)
            
            if not sentences:
                continue
                
            all_sentences.extend(sentences)
            doc_sentence_counts.append(len(sentences))
            # Chuyển row thành dict để tránh tràn RAM khi lưu trữ
            doc_metadata.append((row.to_dict(), sentences))
            
        t_split = time.time()
        
        if not all_sentences:
            continue
            
        # 2. Encode Sentences (KHÔNG NỐI CHUỖI -> Tốc độ x3)
        # Chỉ encode đúng N câu, không bắt GPU làm x3 lần
        print(f"\n[Batch] Encoding {len(all_sentences)} sentences...")
        embeddings = model.encode(
            all_sentences, 
            batch_size=batch_size, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        t_encode = time.time()
        
        # 3. Tính Moving Average Vector & Chunking
        curr_idx = 0
        for doc_idx, count in enumerate(doc_sentence_counts):
            row_dict, original_sentences = doc_metadata[doc_idx]
            
            # Lấy vector của riêng document này
            doc_embeddings = embeddings[curr_idx : curr_idx+count]
            curr_idx += count
            
            if count <= 1:
                row_dict['content'] = original_sentences[0]
                row_dict['chunk_id'] = 0
                row_dict['original_doc_id'] = row_dict.get('doc_id')
                row_dict['doc_id'] = f"{row_dict.get('doc_id')}_chunk0"
                new_rows.append(row_dict)
                continue
                
            # --- Moving Average Context (Lấy context mà không tốn GPU) ---
            # Mỗi câu được tính trung bình với câu trước và câu sau (Window = 3)
            context_embeddings = np.copy(doc_embeddings)
            for i in range(count):
                start = max(0, i - 1)
                end = min(count, i + 2)
                avg_vec = np.mean(doc_embeddings[start:end], axis=0)
                # Chuẩn hóa lại vector (L2 normalize)
                norm = np.linalg.norm(avg_vec)
                context_embeddings[i] = avg_vec / norm if norm > 0 else avg_vec
                
            # Tính khoảng cách (1 - Cosine Similarity) giữa các vector context
            distances = []
            for i in range(count - 1):
                sim = np.dot(context_embeddings[i], context_embeddings[i+1])
                distances.append(1.0 - sim)
                
            # --- Tính Ngưỡng Cắt (Dynamic kết hợp Absolute Min) ---
            # Ngưỡng = Mean + Standard Deviation. Nhưng KHÔNG được nhỏ hơn 0.3
            # BGE-Large: distance 0.3 <=> cosine sim 0.7. Độ tương đồng < 0.7 mới cho phép cắt.
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            absolute_min_dist = 0.3 
            
            threshold = max(absolute_min_dist, mean_dist + std_dist)
            
            # --- Bắt đầu cắt ---
            chunks = []
            current_chunk = [original_sentences[0]]
            
            for i, dist in enumerate(distances):
                if dist > threshold:
                    # Gãy Topic -> Đóng chunk cũ, tạo chunk mới
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [original_sentences[i+1]]
                else:
                    current_chunk.append(original_sentences[i+1])
                    
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            # Lưu lại vào new_rows
            for c_idx, chunk_text in enumerate(chunks):
                new_chunk_dict = row_dict.copy()
                new_chunk_dict['content'] = chunk_text
                new_chunk_dict['chunk_id'] = c_idx
                new_chunk_dict['original_doc_id'] = row_dict.get('doc_id')
                new_chunk_dict['doc_id'] = f"{row_dict.get('doc_id')}_chunk{c_idx}"
                new_rows.append(new_chunk_dict)
                
        t_end = time.time()
        
        print(f"Timing - Split: {t_split - t_start:.2f}s | Encode: {t_encode - t_split:.2f}s | Chunk Math: {t_end - t_encode:.2f}s | Total: {t_end - t_start:.2f}s")
                
    chunked_df = pd.DataFrame(new_rows)
    print(f"\nOriginal documents: {len(df)}")
    print(f"Chunked documents: {len(chunked_df)}")
    
    print(f"Saving chunked data to {output_file}")
    chunked_df.to_parquet(output_file)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=1024, help="GPU batch size for encoding")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.embedding_model, args.batch_size)
