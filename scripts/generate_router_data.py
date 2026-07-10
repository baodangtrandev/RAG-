import os
import argparse
import logging
import pandas as pd
import requests
import json
import uuid
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_BASE_URL = os.getenv("LLM_API_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")

if not API_BASE_URL or not API_KEY:
    raise ValueError("Missing API configuration. Please check your .env file.")

ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_queries_for_chunk(chunk_content: str, source_type: str, num_queries: int = 2) -> list[dict]:
    """
    Gọi GPT-4o qua API Proxy để sinh ra các câu hỏi giả lập dựa trên nội dung tài liệu.
    """
    prompt = (
        f"You are an expert data generator for an Enterprise RAG system. "
        f"I will provide you with a source document extracted from: '{source_type}'.\n\n"
        f"Your task is to role-play as a company employee and generate {num_queries} realistic, practical questions "
        f"whose answers can be directly found within this document.\n"
        f"For each question, you MUST also provide the 'gold_answer' (a comprehensive answer based on the document) "
        f"and 'answer_facts' (a list of atomic facts extracted from the gold_answer).\n\n"
        f"Please return ONLY a valid JSON array of objects. Do not include markdown formatting blocks like ```json.\n"
        f"Each object must strictly follow this JSON schema:\n"
        f"{{\n"
        f"  \"question\": \"The generated question\",\n"
        f"  \"gold_answer\": \"The comprehensive answer\",\n"
        f"  \"answer_facts\": [\"Fact 1\", \"Fact 2\"]\n"
        f"}}\n\n"
        f"--- DOCUMENT ---\n{chunk_content}"
    )
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful Enterprise assistant. Please respond with valid JSON arrays only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=40)
        if response.status_code == 200:
            data = response.json()
            reply = data['choices'][0]['message']['content'].strip()
            
            # Clean up potential markdown formatting
            if reply.startswith('```json'):
                reply = reply[7:]
            if reply.startswith('```'):
                reply = reply[3:]
            if reply.endswith('```'):
                reply = reply[:-3]
                
            parsed_data = json.loads(reply.strip())
            if isinstance(parsed_data, list):
                return parsed_data
            else:
                logger.error("API did not return a JSON array.")
                return []
        else:
            logger.error(f"API Error (Status {response.status_code}): {response.text}")
            return []
    except Exception as e:
        logger.error(f"Request/Parsing failed: {str(e)}")
        return []

def worker(row) -> list[dict]:
    """
    Hàm worker cho ThreadPool, xử lý 1 chunk và trả về list các dictionary chứa câu hỏi.
    """
    source_type = str(row.get('source_type', ''))
    doc_id = str(row.get('doc_id', ''))
    content = str(row.get('content', ''))
    
    queries_data = generate_queries_for_chunk(content, source_type, num_queries=2)
    
    results = []
    for item in queries_data:
        results.append({
            "question_id": f"qst_{uuid.uuid4().hex[:8]}",
            "question_type": "basic",
            "source_types": [source_type],
            "question": item.get("question", ""),
            "expected_doc_ids": [doc_id],
            "gold_answer": item.get("gold_answer", ""),
            "answer_facts": item.get("answer_facts", [])
        })
    return results

def main(input_file: str, output_file: str, samples_per_source: int, max_workers: int):
    logger.info(f"Đọc dữ liệu từ: {input_file}")
    df = pd.read_parquet(input_file)
    
    source_types = df['source_type'].unique()
    logger.info(f"Tìm thấy {len(source_types)} nguồn dữ liệu: {source_types}")
    
    sampled_df = pd.DataFrame()
    for src in source_types:
        subset = df[df['source_type'] == src]
        n_samples = min(samples_per_source, len(subset))
        sampled = subset.sample(n=n_samples, random_state=42)
        sampled_df = pd.concat([sampled_df, sampled])
        
    logger.info(f"Đã chọn ra tổng cộng {len(sampled_df)} document để đưa cho AI sinh câu hỏi.")
    
    records = sampled_df.to_dict('records')
    training_data = []
    
    logger.info(f"Bắt đầu gọi API tới GPT-4o với {max_workers} luồng...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, row): row for row in records}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Queries"):
            result = future.result()
            if result:
                training_data.extend(result)
                
    if not training_data:
        logger.error("Quá trình sinh dữ liệu thất bại hoặc trả về 0 kết quả.")
        return
        
    # Đảm bảo lưu đúng định dạng JSONL như tập test mẫu
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in training_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info(f"Đã sinh thành công {len(training_data)} câu hỏi huấn luyện!")
    logger.info(f"Dữ liệu JSONL đã được lưu chuẩn xác vào: {output_file}")
    
    print("\n--- [Preview Data] ---")
    print(json.dumps(training_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Router Data Generation")
    parser.add_argument("--input-file", type=str, default="data/EnterpriseRAG-Bench/data/documents/test.parquet")
    parser.add_argument("--output-file", type=str, default="data/router_training_data.jsonl")
    parser.add_argument("--samples-per-source", type=int, default=150, help="Số lượng chunks lấy từ mỗi nguồn để sinh câu hỏi")
    parser.add_argument("--max-workers", type=int, default=10, help="Số luồng gọi API song song (Tăng lên nếu API mạnh)")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.samples_per_source, args.max_workers)
