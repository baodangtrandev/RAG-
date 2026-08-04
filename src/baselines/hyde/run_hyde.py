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
logger = logging.getLogger("hyde_baseline")

app = typer.Typer(name="hyde-baseline")


@app.command()
def run(
    questions_file: str = typer.Option("data/EnterpriseRAG-Bench/data/questions/test.parquet", "--questions", "-q"),
    output_file: str = typer.Option("results/baseline_hyde.jsonl", "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
):
    """Run HyDE (Hypothetical Document Embeddings) + Hybrid Search + Reranker Baseline"""
    logger.info("=" * 60)
    logger.info("HyDE BASELINE")
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

    # --- Phase 1: Sinh hypothetical documents bằng LLM ---
    logger.info("Generating hypothetical documents for HyDE...")
    hyde_prompts = []
    for q in queries:
        prompt = (
            f"<|im_start|>system\nYou are a precise enterprise assistant. Write a short paragraph answering "
            f"the user's question. This answer will be used to retrieve documents, so focus on relevant terms and facts.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        )
        hyde_prompts.append(prompt)

    t_gen_start = time.perf_counter()
    from vllm import SamplingParams

    # SamplingParams ngắn hơn cho HyDE
    hyde_params = SamplingParams(temperature=0.7, max_tokens=192, stop=["<|im_end|>", "<|endoftext|>"])
    outputs = generator.llm.generate(hyde_prompts, hyde_params)
    hypo_docs = [out.outputs[0].text.strip() for out in outputs]
    logger.info(f"Hypothetical docs generated in {time.perf_counter() - t_gen_start:.2f}s.")

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

    logger.info("Bắt đầu Hybrid Search trên các hypothetical documents...")
    table_names = db.table_names()
    tables = [db.open_table(name) for name in table_names]

    all_docs = []
    per_query_retrieval_times = []

    for idx, q in enumerate(queries):
        t_q = time.perf_counter()
        hypo_doc = hypo_docs[idx]

        # Embed hypothetical document thay vì query gốc
        emb = router.encoder.encode([hypo_doc], normalize_embeddings=True)[0]

        dense_candidates = []
        sparse_candidates = []

        for t_idx, table in enumerate(tables):
            try:
                res_dense = table.search(emb).limit(top_k_retrieve).to_list()
                for r in res_dense:
                    r["_source"] = table_names[t_idx]
                    dense_candidates.append(r)
            except Exception:
                pass

            try:
                # Sparse search vẫn giữ nguyên query gốc (hoặc hypo_doc, HyDE gốc dùng vector search.
                # Chúng ta chạy hybrid thì sparse search dùng query gốc cho chính xác)
                res_sparse = table.search(q, query_type="fts").limit(top_k_retrieve).to_list()
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

            rrf_docs.append({"content": doc.get("content", ""), "rrf_score": rrf_score})

        rrf_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_k_initial = rrf_docs[:top_k_retrieve]

        if not top_k_initial:
            all_docs.append([])
            per_query_retrieval_times.append(time.perf_counter() - t_q)
            continue

        # Rerank bằng QUERY GỐC để đảm bảo tính đúng đắn và công bằng
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
