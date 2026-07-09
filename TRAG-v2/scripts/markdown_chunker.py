import os
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import re

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install langchain-text-splitters")
    exit(1)

# 1. Cấu hình Markdown Splitter (Nhận diện tiêu đề)
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False # Giữ lại header trong nội dung chunk
)

# 2. Cấu hình Recursive Character (Cắt các đoạn quá dài, giữ nguyên paragraph)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    # Ưu tiên: Đoạn kép -> List -> Dòng đơn -> DẤU CHẤM CÂU -> Khoảng trắng
    separators=["\n\n", "\n- ", "\n* ", "\n", ". ", " ", ""] 
)

def preprocess_text(text, source_type):
    """
    Tiền xử lý văn bản thô (pseudo-markdown) thành chuẩn Markdown
    để MarkdownHeaderTextSplitter có thể nhận diện được.
    Chiến lược Càn Quét: Áp dụng cho mọi nguồn (trừ chat log).
    """
    if not text:
        return text
        
    source_type = str(source_type).lower().strip()
    
    # Slack là hội thoại chat liên tục, tự động ngắt bằng dấu xuống dòng là chuẩn.
    if source_type != 'slack':
        # Bắt các dòng dạng "Từ khóa:" (như Summary:, Goals:, description:)
        text = re.sub(r'(?m)^([A-Za-z0-9][A-Za-z0-9_ \-]{2,80}):\s*$', r'## \1:', text)
        
        # Bắt các dòng header email (From:, To:, Subject:, Date:, Cc:)
        text = re.sub(r'(?m)^(From|To|Subject|Date|Cc):\s*', r'## \1: ', text)
        
        # Bắt các tiêu đề thô ngắn (viết hoa chữ đầu, đứng 1 mình 1 dòng, không có dấu chấm/phẩy ở cuối)
        text = re.sub(r'(?m)^([A-Z][a-zA-Z0-9 &()_\-]{2,80})$', r'## \1', text)
    
    return text

def process_row(row_dict):
    """
    Hàm xử lý cho một document duy nhất. 
    Hàm này phải đứng độc lập để chạy đa luồng (multiprocessing).
    """
    content = str(row_dict.get('content', '')) if pd.notna(row_dict.get('content')) else ""
    title = str(row_dict.get('title', '')) if 'title' in row_dict and pd.notna(row_dict.get('title')) else ""
    original_doc_id = str(row_dict.get('doc_id'))
    source_type = str(row_dict.get('source_type', ''))
    
    if not content.strip():
        return []
        
    # --- BƯỚC 1: TIỀN XỬ LÝ (BIẾN HÌNH MARKDOWN) ---
    content = preprocess_text(content, source_type)
    
    # Biến Title gốc thành Header 1 để luôn được nhận diện và không bao giờ bị cắt mất
    full_content = f"# Title: {title}\n\n{content}" if title.strip() else content
    
    try:
        # --- BƯỚC 2: CẮT THEO CẤU TRÚC MARKDOWN ---
        md_docs = MARKDOWN_SPLITTER.split_text(full_content)
        # Những đoạn nào lọt qua mà vẫn dài > 1000 ký tự thì RecursiveSplitter sẽ chia nhỏ tiếp
        final_chunks = TEXT_SPLITTER.split_documents(md_docs)
    except Exception:
        # Fallback an toàn nếu văn bản có ký tự lạ làm lỗi Markdown Splitter
        final_chunks = TEXT_SPLITTER.create_documents([full_content])
        
    results = []
    for c_idx, doc in enumerate(final_chunks):
        new_row = row_dict.copy()
        
        # Lấy ngữ cảnh Header (Ví dụ: "Header 1: Title > Header 2: Description")
        header_vals = [str(v) for k, v in doc.metadata.items() if k.startswith("Header")]
        header_context = " > ".join(header_vals)
        
        chunk_text = doc.page_content.strip()
        
        # --- BƯỚC 3: MẸO PRO CHO RAG PRODUCTION ---
        # Nếu Recursive Splitter cắt đôi 1 mục có Header, nửa sau sẽ bị mồ côi (không chứa dấu #).
        # Ta sẽ tự động tiêm (inject) Header gốc của nó vào đầu chunk để LLM/Embedding Model không bị mất ngữ cảnh.
        if header_context and not chunk_text.lstrip().startswith("#"):
            chunk_text = f"[{header_context}]\n{chunk_text}"
            
        new_row['content'] = chunk_text
        new_row['chunk_id'] = c_idx
        new_row['original_doc_id'] = original_doc_id
        new_row['doc_id'] = f"{original_doc_id}_chunk{c_idx}"
        
        results.append(new_row)
        
    return results

def main(input_file, output_file, max_workers):
    print(f"Đang tải dữ liệu từ {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Fix array objects in pandas (nếu có)
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
            
    print(f"🚀 Bắt đầu Markdown Chunking (kèm Tiền xử lý Regex càn quét) {len(df)} documents với {max_workers} CPU cores...")
    
    # Chuyển đổi sang list dictionary để truyền vào các worker processes
    rows = df.to_dict('records')
    all_chunks = []
    
    # Sử dụng đa luồng (Multiprocessing) để tận dụng sức mạnh CPU -> Chạy cực nhanh
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Dùng chunksize lớn (100) để giảm overhead của đa luồng
        results = list(tqdm(executor.map(process_row, rows, chunksize=100), total=len(rows), desc="Chunking Tiến Độ"))
        
    for res in results:
        all_chunks.extend(res)
            
    chunked_df = pd.DataFrame(all_chunks)
    
    print(f"\n✅ HOÀN THÀNH!")
    print(f"Số document gốc: {len(df)}")
    print(f"Số chunk tạo ra: {len(chunked_df)}")
    print(f"Tỉ lệ cắt trung bình: {len(chunked_df)/len(df):.2f} chunks / document")
    
    print(f"\nĐang lưu kết quả ra file: {output_file}")
    chunked_df.to_parquet(output_file)
    print("Xong!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    # Tự động lấy tối đa số nhân CPU trừ 1 để máy không bị đơ
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.workers)
