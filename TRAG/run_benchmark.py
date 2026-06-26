import json
import time
from vllm_engine import TRAG_LLMEngine
from trag_pipeline import TRAGPipeline

QUESTIONS_FILE = "../EnterpriseRAG-Bench/questions.jsonl"
ANSWERS_FILE = "../EnterpriseRAG-Bench/answers.jsonl"
DB_PATH = "../lancedb_data"

def run_benchmark():
    # Load questions
    questions_data = []
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions_data.append(json.loads(line))
                
    questions = [q["question"] for q in questions_data]
    q_ids = [q["question_id"] for q in questions_data]
    
    print(f"Loaded {len(questions)} questions. Initializing Engines...")
    
    # Initialize Engine & Pipeline
    llm = TRAG_LLMEngine(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    pipeline = TRAGPipeline(db_path=DB_PATH, llm_engine=llm)
    
    # Run Batch Stages
    start_time = time.time()
    
    parsed_queries = pipeline.batch_parse_queries(questions)
    retrieved_docs = pipeline.batch_hybrid_retrieval(questions, parsed_queries, top_k=50)
    top_k_docs = pipeline.batch_temporal_reranking(questions, retrieved_docs, top_k=10, lambda_val=0.0)
    answers = pipeline.batch_generate_answers(questions, top_k_docs)
    
    end_time = time.time()
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")
    print(f"Throughput: {len(questions) / (end_time - start_time):.2f} queries/sec")
    
    # Write to answers.jsonl
    with open(ANSWERS_FILE, 'w', encoding='utf-8') as f:
        for i in range(len(questions)):
            # extract document ids
            doc_ids = [doc.get("doc_id") for doc in top_k_docs[i]]
            ans_obj = {
                "question_id": q_ids[i],
                "answer": answers[i],
                "document_ids": doc_ids
            }
            f.write(json.dumps(ans_obj) + "\n")
            
    print(f"Results saved to {ANSWERS_FILE}")

if __name__ == "__main__":
    run_benchmark()
