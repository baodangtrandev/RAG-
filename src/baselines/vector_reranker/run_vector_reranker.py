import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("vector_reranker_baseline")

app = typer.Typer(name="vector-reranker-baseline")

@app.command()
def run(
    questions_file: str = typer.Option("data/EnterpriseRAG-Bench/data/questions/test.parquet", "--questions", "-q"),
    output_file: str = typer.Option("results/baseline_vector_reranker.jsonl", "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
):
    """Run Vector Search + Reranker Baseline"""
    logger.info("=" * 60)
    logger.info("VECTOR SEARCH + RERANKER BASELINE")
    logger.info("=" * 60)

    from src.generation.generator import VLLMGenerator
    logger.info("Initializing vLLM Generator...")
    generator = VLLMGenerator()

    pipeline_start = time.perf_counter()

    qpath = Path(questions_file)
    queries, question_ids = [], []
    if qpath.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(questions_file)
        queries = df["question"].tolist()
        question_ids = df.index.tolist() if "question_id" not in df.columns else df["question_id"].tolist()
    elif qpath.suffix in (".jsonl", ".json"):
        with open(questions_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                queries.append(item["question"])
                question_ids.append(item.get("question_id", idx))

    if limit:
        queries = queries[:limit]
        question_ids = question_ids[:limit]

    import lancedb
    import numpy as np
    from src.models.router_inference import ProbabilisticSourceRouter
    from src.reranker.reranker import CrossEncoderReranker
    
    router = ProbabilisticSourceRouter(model_dir=os.environ.get("PSR_MODEL_DIR", "models/psr_v2"))
    db = lancedb.connect(os.environ.get("RAG_DB_URI", "data/lancedb"))
    reranker = CrossEncoderReranker()
    
    top_k_retrieve = int(os.environ.get("RAG_TOP_K_RETRIEVE", 20))
    top_k_final = int(os.environ.get("RAG_TOP_K_FINAL", 5))
    
    logger.info("Bắt đầu Vector Search + Reranking...")
    table_names = db.table_names()
    tables = [db.open_table(name) for name in table_names]
    
    all_docs = []
    per_query_retrieval_times = []
    for q in queries:
        t_q = time.perf_counter()
        emb = router.encoder.encode([q], normalize_embeddings=True)[0]
        q_results = []
        for table in tables:
            try:
                res = table.search(emb).limit(top_k_retrieve).to_list()
                for r in res:
                    q_results.append({
                        "content": r.get("content", ""),
                        "distance": r.get("_distance", 1.0)
                    })
            except Exception as e:
                pass
        
        q_results.sort(key=lambda x: x["distance"])
        top_k_initial = [doc for doc in q_results[:top_k_retrieve]]
        
        # Rerank phase
        if not top_k_initial:
            all_docs.append([])
            per_query_retrieval_times.append(time.perf_counter() - t_q)
            continue
            
        pairs = [[q, doc["content"]] for doc in top_k_initial]
        scores = reranker.model.predict(pairs)
        
        for doc, score in zip(top_k_initial, scores):
            doc["rerank_score"] = float(score)
            
        top_k_initial.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_final_docs = [doc for doc in top_k_initial[:top_k_final]]
        all_docs.append(top_final_docs)
        per_query_retrieval_times.append(time.perf_counter() - t_q)

    answers = generator.generate_batch(queries, all_docs)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_elapsed = time.perf_counter() - pipeline_start
    avg_latency = total_elapsed / max(len(queries), 1)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (qid, query, answer) in enumerate(zip(question_ids, queries, answers)):
            record = {
                "question_id": str(qid),
                "question": query,
                "answer": answer,
                "latency_sec": round(avg_latency, 4),
                "retrieval_latency_sec": round(per_query_retrieval_times[idx], 4),
                "search_space_docs": 4213106
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. Avg latency: %.4fs", avg_latency)

if __name__ == "__main__":
    app()
