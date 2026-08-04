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
logger = logging.getLogger("bm25_baseline")

app = typer.Typer(name="bm25-baseline")


@app.command()
def run(
    questions_file: str = typer.Option("data/EnterpriseRAG-Bench/data/questions/test.parquet", "--questions", "-q"),
    output_file: str = typer.Option("results/baseline_bm25.jsonl", "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
):
    """Run BM25 Baseline Pipeline"""
    logger.info("=" * 60)
    logger.info("BM25 BASELINE")
    logger.info("=" * 60)

    from src.generation.generator import VLLMGenerator

    logger.info("Initializing vLLM Generator...")
    generator = VLLMGenerator()

    pipeline_start = time.perf_counter()

    # Load questions
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

    # Khởi tạo BM25 Retriever (Dùng tính năng Full Text Search của LanceDB)
    import lancedb

    db = lancedb.connect(os.environ.get("RAG_DB_URI", "data/lancedb"))

    top_k_retrieve = int(os.environ.get("RAG_TOP_K_RETRIEVE", 20))
    top_k_final = int(os.environ.get("RAG_TOP_K_FINAL", 5))

    logger.info("Bắt đầu BM25 (FTS) Baseline...")
    table_names = db.table_names()
    tables = []
    for name in table_names:
        table = db.open_table(name)
        # Tự động tạo FTS Index nếu chưa có
        try:
            table.create_fts_index("content", replace=False)
            logger.info(f"  FTS Index sẵn sàng cho bảng: {name}")
        except Exception as e:
            logger.warning(f"  Không thể tạo FTS Index cho bảng {name}: {e}")
        tables.append(table)

    all_docs = []
    per_query_retrieval_times = []
    for q_idx, q in enumerate(queries):
        t_q = time.perf_counter()
        q_results = []
        for t_idx, table in enumerate(tables):
            try:
                # Sử dụng query_type="fts" để kích hoạt Full Text Search (BM25/Tantivy)
                # thay vì Vector Search mặc định
                res = table.search(q, query_type="fts").limit(top_k_retrieve).to_list()
                for r in res:
                    q_results.append(
                        {
                            "content": r.get("content", ""),
                            "title": r.get("title", ""),
                            "source": table_names[t_idx],
                            "score": r.get("_score", r.get("score", 0.0)),
                        }
                    )
            except Exception as e:
                if q_idx == 0:  # Chỉ log lần đầu để tránh spam
                    logger.warning(f"  BM25 search lỗi ở bảng {table_names[t_idx]}: {e}")
        # Sắp xếp lại theo điểm BM25 giảm dần
        q_results.sort(key=lambda x: x["score"], reverse=True)
        # Lấy Top K tài liệu chung cuộc theo file env
        top_final_docs = [doc for doc in q_results[:top_k_final]]
        all_docs.append(top_final_docs)
        per_query_retrieval_times.append(time.perf_counter() - t_q)

    answers = generator.generate_batch(queries, all_docs)

    # Save output
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
