import json
import logging
import os
import sys
import time
import re
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("llm_router_baseline")

app = typer.Typer(name="llm-router-baseline")

@app.command()
def run(
    questions_file: str = typer.Option("data/EnterpriseRAG-Bench/data/questions/test.parquet", "--questions", "-q"),
    output_file: str = typer.Option("results/baseline_llm_router.jsonl", "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
):
    """Run LLM-Router (Agentic Routing) + Hybrid Search + Reranker Baseline"""
    logger.info("=" * 60)
    logger.info("LLM ROUTER BASELINE")
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

    # --- Phase 1: Định tuyến bằng LLM ---
    logger.info("Routing queries using LLM (Agentic)...")
    llm_router_prompts = []
    for q in queries:
        prompt = (
            f"<|im_start|>system\nYou are a database router for an enterprise RAG. Select which of the "
            f"following databases contain relevant information for the user's query:\n"
            f"- confluence: wiki pages, guides, project descriptions, documentations\n"
            f"- jira: tickets, bug tracking, user stories, status, tasks\n"
            f"- github: source code, pull requests, commits, code issues\n"
            f"- slack: chat messages, team conversations\n"
            f"- gmail: email communications\n"
            f"- hubspot: sales, customer crm, marketing data\n"
            f"- linear: project planning, tasks\n"
            f"- google_drive: file assets, spreadsheets, reports\n"
            f"- fireflies: meeting transcripts, audio notes\n"
            f"Return ONLY a comma-separated list of relevant databases (e.g. 'jira, slack'). "
            f"If not sure, write 'all'. Do not write explanations.<|im_end|>\n"
            f"<|im_start|>user\nQuery: {q}<|im_end|>\n<|im_start|>assistant\n"
        )
        llm_router_prompts.append(prompt)

    t_gen_start = time.perf_counter()
    from vllm import SamplingParams
    router_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    outputs = generator.llm.generate(llm_router_prompts, router_params)
    
    valid_tables = {'confluence', 'fireflies', 'github', 'gmail', 'google_drive', 'hubspot', 'jira', 'linear', 'slack'}
    active_shards_list = []
    
    for out in outputs:
        text = out.outputs[0].text.strip().lower()
        if 'all' in text or not text:
            active_shards_list.append(list(valid_tables))
        else:
            found = []
            for word in re.findall(r'[a-z\_]+', text):
                if word in valid_tables:
                    found.append(word)
            if not found:
                found = list(valid_tables)
            active_shards_list.append(found)
            
    logger.info(f"LLM routing done in {time.perf_counter() - t_gen_start:.2f}s.")

    # --- Phase 2: Retrieval ---
    import lancedb
    from src.models.router_inference import ProbabilisticSourceRouter
    from src.reranker.reranker import CrossEncoderReranker

    router = ProbabilisticSourceRouter(model_dir=os.environ.get("PSR_MODEL_DIR", "models/psr_v2"))
    db = lancedb.connect(os.environ.get("RAG_DB_URI", "data/lancedb"))
    reranker = CrossEncoderReranker()

    top_k_retrieve = int(os.environ.get("RAG_TOP_K_RETRIEVE", 20))
    top_k_final = int(os.environ.get("RAG_TOP_K_FINAL", 5))
    k_rrf = 60

    logger.info("Bắt đầu Hybrid Search trên các shards được LLM chỉ định...")
    all_docs = []
    per_query_retrieval_times = []
    search_spaces = []

    for idx, q in enumerate(queries):
        t_q = time.perf_counter()
        emb = router.encoder.encode([q], normalize_embeddings=True)[0]
        
        shards = active_shards_list[idx]
        
        # Tính toán Search Space (tổng dung lượng các bảng được quét)
        # Giả lập table size giống retriever_v2.py để so sánh chuẩn
        # Kích thước ước lượng của mỗi table:
        table_sizes = {
            "confluence": 1064736,
            "fireflies": 673813,
            "github": 659802,
            "gmail": 254823,
            "google_drive": 348712,
            "hubspot": 105824,
            "jira": 659802,
            "linear": 254823,
            "slack": 190771
        }
        search_space = sum(table_sizes.get(s, 0) for s in shards)
        search_spaces.append(search_space)

        dense_candidates = []
        sparse_candidates = []
        
        for name in shards:
            try:
                table = db.open_table(name)
                res_dense = table.search(emb).limit(top_k_retrieve).to_list()
                for r in res_dense:
                    r["_source"] = name
                    dense_candidates.append(r)
            except Exception:
                pass
                
            try:
                table = db.open_table(name)
                res_sparse = table.search(q, query_type="fts").limit(top_k_retrieve).to_list()
                for r in res_sparse:
                    r["_source"] = name
                    sparse_candidates.append(r)
            except Exception:
                pass

        # Global RRF Sorting
        dense_candidates.sort(key=lambda x: x.get("_distance", float('inf')))
        sparse_candidates.sort(key=lambda x: x.get("_score", x.get("score", 0.0)), reverse=True)
        
        fused_docs = {}
        for rank_idx, doc in enumerate(dense_candidates):
            key = (doc["_source"], doc.get("doc_id", "unknown"))
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": rank_idx + 1, "sparse_rank": None}
            else:
                fused_docs[key]["dense_rank"] = rank_idx + 1
                 
        for rank_idx, doc in enumerate(sparse_candidates):
            key = (doc["_source"], doc.get("doc_id", "unknown"))
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": None, "sparse_rank": rank_idx + 1}
            else:
                fused_docs[key]["sparse_rank"] = rank_idx + 1
                 
        rrf_docs = []
        for key, info in fused_docs.items():
            doc = info["doc"]
            dense_rank = info["dense_rank"]
            sparse_rank = info["sparse_rank"]
            
            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += 0.5 * (1.0 / (k_rrf + dense_rank))
            if sparse_rank is not None:
                rrf_score += 0.5 * (1.0 / (k_rrf + sparse_rank))
                
            rrf_docs.append({
                "content": doc.get("content", ""),
                "rrf_score": rrf_score
            })
            
        rrf_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_k_initial = rrf_docs[:top_k_retrieve]
        
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

    # Sinh câu trả lời cuối cùng
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
                "search_space_docs": int(search_spaces[idx])
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. Avg latency: %.4fs", avg_latency)

if __name__ == "__main__":
    app()
