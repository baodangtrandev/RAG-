import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("query_expansion_baseline")

app = typer.Typer(name="query-expansion-baseline")


@app.command()
def run(
    questions_file: str = typer.Option("data/EnterpriseRAG-Bench/data/questions/test.parquet", "--questions", "-q"),
    output_file: str = typer.Option("results/baseline_query_expansion.jsonl", "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
):
    """Run Multi-Query Expansion + Hybrid Search + Reranker Baseline"""
    logger.info("=" * 60)
    logger.info("QUERY EXPANSION BASELINE")
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

    # --- Phase 1: Tạo các query mở rộng bằng LLM ---
    logger.info("Generating expanded queries...")
    qe_prompts = []
    for q in queries:
        prompt = (
            f"<|im_start|>system\nYou are a helpful search assistant. Generate exactly 3 different search queries "
            f"to retrieve documents for the user's question. Focus on synonyms, keywords, and alternative ways "
            f"to ask the question. Write exactly ONE query per line. Do not write numbers, bullet points or explanations.<|im_end|>\n"
            f"<|im_start|>user\nUser question: {q}<|im_end|>\n<|im_start|>assistant\n"
        )
        qe_prompts.append(prompt)

    t_gen_start = time.perf_counter()
    from vllm import SamplingParams

    qe_params = SamplingParams(temperature=0.5, max_tokens=128, stop=["<|im_end|>", "<|endoftext|>"])
    outputs = generator.llm.generate(qe_prompts, qe_params)

    expanded_queries_list = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_lines = []
        for line in lines:
            # Clean up potential leading numbers or bullets
            line = re.sub(r"^\d+[\.\-\)]\s*", "", line)
            line = re.sub(r"^[\-\*\•]\s*", "", line)
            if line:
                clean_lines.append(line)
        expanded_queries_list.append(clean_lines[:3])
    logger.info(f"Query expansion generated in {time.perf_counter() - t_gen_start:.2f}s.")

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

    logger.info("Bắt đầu Multi-Query Hybrid Search...")
    table_names = db.table_names()
    tables = [db.open_table(name) for name in table_names]

    all_docs = []
    per_query_retrieval_times = []

    for idx, q in enumerate(queries):
        t_q = time.perf_counter()

        # Tập hợp tất cả các query cần chạy (gốc + 3 mở rộng)
        sub_queries = [q] + expanded_queries_list[idx]

        dense_candidates = []
        sparse_candidates = []

        # Chạy tìm kiếm cho từng sub_query
        for sq in sub_queries:
            emb = router.encoder.encode([sq], normalize_embeddings=True)[0]

            for t_idx, table in enumerate(tables):
                try:
                    res_dense = table.search(emb).limit(top_k_retrieve).to_list()
                    for r in res_dense:
                        r["_source"] = table_names[t_idx]
                        dense_candidates.append(r)
                except Exception:
                    pass

                try:
                    res_sparse = table.search(sq, query_type="fts").limit(top_k_retrieve).to_list()
                    for r in res_sparse:
                        r["_source"] = table_names[t_idx]
                        sparse_candidates.append(r)
                except Exception:
                    pass

        # Global RRF Sorting
        dense_candidates.sort(key=lambda x: x.get("_distance", float("inf")))
        sparse_candidates.sort(key=lambda x: x.get("_score", x.get("score", 0.0)), reverse=True)

        fused_docs = {}
        for rank_idx, doc in enumerate(dense_candidates):
            key = (doc["_source"], doc.get("doc_id", "unknown"))
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": rank_idx + 1, "sparse_rank": None}
            else:
                # Nếu đã có, giữ rank nhỏ hơn (tức là hạng tốt hơn từ sub-query tốt hơn)
                if fused_docs[key]["dense_rank"] is None or rank_idx + 1 < fused_docs[key]["dense_rank"]:
                    fused_docs[key]["dense_rank"] = rank_idx + 1

        for rank_idx, doc in enumerate(sparse_candidates):
            key = (doc["_source"], doc.get("doc_id", "unknown"))
            if key not in fused_docs:
                fused_docs[key] = {"doc": doc, "dense_rank": None, "sparse_rank": rank_idx + 1}
            else:
                if fused_docs[key]["sparse_rank"] is None or rank_idx + 1 < fused_docs[key]["sparse_rank"]:
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

            rrf_docs.append({"content": doc.get("content", ""), "rrf_score": rrf_score})

        rrf_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_k_initial = rrf_docs[:top_k_retrieve]

        if not top_k_initial:
            all_docs.append([])
            per_query_retrieval_times.append(time.perf_counter() - t_q)
            continue

        # Rerank bằng QUERY GỐC
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
                "search_space_docs": 4213106,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. Avg latency: %.4fs", avg_latency)


if __name__ == "__main__":
    app()
