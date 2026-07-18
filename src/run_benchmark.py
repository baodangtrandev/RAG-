"""
T-RAG Benchmark Runner
=======================
Script chinh ket noi toan bo pipeline theo dang Stage-based Batching
de toi da hoa Throughput tren GPU H100.

Su dung:
    # Chay voi config mac dinh tu .env
    python src/run_benchmark.py

    # Ghi de model de benchmark so sanh
    python src/run_benchmark.py --model Qwen/Qwen2.5-14B-Instruct

    # Ghi de nhieu tham so cung luc
    python src/run_benchmark.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --tau 0.2 --top-k-final 3

Luong xu ly (Stage-based):
    Stage 1: Load questions dataset
    Stage 2: Batch CSEP Retrieval (Hop 1 -> Entity Extraction -> Hop 2)
    Stage 3: Batch Cross-Encoder Reranking
    Stage 4: Batch LLM Generation (vLLM PagedAttention)
    Stage 5: Ghi ket qua ra answers.jsonl
"""

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

# Them project root vao sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cau hinh logging chuan
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trag.benchmark")

app = typer.Typer(
    name="trag-benchmark",
    help="T-RAG Pipeline Benchmark Runner — Stage-based Batching on H100",
    add_completion=False,
)


def _load_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _load_env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@app.command()
def run(
    questions_file: str = typer.Option(
        "data/EnterpriseRAG-Bench/data/questions/test.parquet",
        "--questions", "-q",
        help="Path toi file chua cac cau hoi (parquet hoac jsonl).",
    ),
    output_file: str = typer.Option(
        "answers.jsonl",
        "--output", "-o",
        help="File ket qua dau ra (JSONL).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="HuggingFace model ID de ghi de LOCAL_LLM_MODEL trong .env.",
    ),
    tau: Optional[float] = typer.Option(
        None,
        "--tau",
        help="Nguong Router tau (ghi de RAG_TAU trong .env).",
    ),
    top_k_retrieve: Optional[int] = typer.Option(
        None,
        "--top-k-retrieve",
        help="So docs lay ra truoc Reranker (ghi de RAG_TOP_K_RETRIEVE).",
    ),
    top_k_final: Optional[int] = typer.Option(
        None,
        "--top-k-final",
        help="So docs dua vao LLM sau Reranker (ghi de RAG_TOP_K_FINAL).",
    ),
    enable_csep: Optional[bool] = typer.Option(
        None,
        "--csep/--no-csep",
        help="Bat/tat CSEP (ghi de ENABLE_CSEP_FOR_ALL trong .env).",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit", "-l",
        help="Gioi han so cau hoi de test nhanh (None = chay tat ca).",
    ),
    no_reranker: bool = typer.Option(
        False,
        "--no-reranker",
        help="Bo qua buoc Reranker de test toc do",
    ),
):
    """
    Chay T-RAG Pipeline tren tap du lieu benchmark va ghi ket qua ra JSONL.
    """
    # --- Ghi de env neu co CLI args ---
    if model:
        os.environ["LOCAL_LLM_MODEL"] = model
    if tau is not None:
        os.environ["RAG_TAU"] = str(tau)
    if top_k_retrieve is not None:
        os.environ["RAG_TOP_K_RETRIEVE"] = str(top_k_retrieve)
    if top_k_final is not None:
        os.environ["RAG_TOP_K_FINAL"] = str(top_k_final)
    if enable_csep is not None:
        os.environ["ENABLE_CSEP_FOR_ALL"] = str(enable_csep)

    # In ra config hien tai
    cfg_model = os.environ.get("LOCAL_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    cfg_tau = os.environ.get("RAG_TAU", "0.15")
    cfg_top_k_r = os.environ.get("RAG_TOP_K_RETRIEVE", "20")
    cfg_top_k_f = os.environ.get("RAG_TOP_K_FINAL", "5")
    cfg_csep = os.environ.get("ENABLE_CSEP_FOR_ALL", "True")

    logger.info("=" * 60)
    logger.info("T-RAG BENCHMARK CONFIG")
    logger.info("  Model            : %s", cfg_model)
    logger.info("  Tau              : %s", cfg_tau)
    logger.info("  top_k_retrieve   : %s", cfg_top_k_r)
    logger.info("  top_k_final      : %s", cfg_top_k_f)
    logger.info("  CSEP             : %s", cfg_csep)
    logger.info("  Questions file   : %s", questions_file)
    logger.info("  Output file      : %s", output_file)
    logger.info("=" * 60)

    # ============================================================
    # Stage 0: Initialize Generator
    # ============================================================
    logger.info("[Stage 0] Initializing vLLM Generator (loads LLM onto GPU)...")
    from src.generation.generator import VLLMGenerator
    generator = VLLMGenerator()

    pipeline_start = time.perf_counter()

    # ============================================================
    # Stage 1: Load Questions Dataset
    # ============================================================
    logger.info("[Stage 1] Loading questions from: %s", questions_file)
    t_s1 = time.perf_counter()

    qpath = Path(questions_file)
    if not qpath.exists():
        logger.error("Questions file not found: %s", questions_file)
        raise typer.Exit(code=1)

    if qpath.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(questions_file)
        queries = df["question"].tolist()
        question_ids = df.index.tolist() if "question_id" not in df.columns else df["question_id"].tolist()
    elif qpath.suffix in (".jsonl", ".json"):
        queries, question_ids = [], []
        with open(questions_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                queries.append(item["question"])
                question_ids.append(item.get("question_id", idx))
    else:
        logger.error("Unsupported file format: %s", qpath.suffix)
        raise typer.Exit(code=1)

    if limit:
        queries = queries[:limit]
        question_ids = question_ids[:limit]

    logger.info("[Stage 1] Loaded %d questions in %.2fs.", len(queries), time.perf_counter() - t_s1)

    # ============================================================
    # Stage 2: Batch CSEP Retrieval
    # ============================================================
    logger.info("[Stage 2] Starting CSEP Retrieval for %d queries...", len(queries))
    t_s2 = time.perf_counter()

    from src.retrieval.csep_retriever import CSEPRetriever

    # Dung ham generate cua generator lam llm_generate_fn cho CSEP
    def csep_llm_fn(prompts):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        import json
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["entities"]
        }
        guided_decoding = GuidedDecodingParams(json=schema)
        entity_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["\n\n", "\nDocuments:", "\nEntities:", "<|im_end|>", "<|endoftext|>", "NONE", ", NONE"],
            guided_decoding=guided_decoding
        )
        outputs = generator.llm.generate(prompts, entity_params)
        return [o.outputs[0].text.strip() for o in outputs]

    csep_retriever = CSEPRetriever(llm_generate_fn=csep_llm_fn)

    # --- Do per-query retrieval latency ---
    per_query_retrieval_times = []
    _original_retrieve = csep_retriever.retriever.retrieve
    def _timed_retrieve(query, top_k=20):
        t0 = time.perf_counter()
        result = _original_retrieve(query, top_k=top_k)
        elapsed = time.perf_counter() - t0
        per_query_retrieval_times.append(elapsed)
        return result
    csep_retriever.retriever.retrieve = _timed_retrieve

    all_docs = csep_retriever.retrieve_batch(queries)
    t_retrieval = time.perf_counter() - t_s2

    logger.info("[Stage 2] Retrieval done in %.2fs.", t_retrieval)

    # ============================================================
    # Stage 3: Batch Cross-Encoder Reranking
    # ============================================================
    if no_reranker:
        logger.info("[Stage 3] SKIPPING Cross-Encoder Reranking (No-Reranker mode).")
        top_k_final_int = int(os.environ.get("RAG_TOP_K_FINAL", "5"))
        reranked_results = []
        for docs in all_docs:
            top_k_docs = docs[:top_k_final_int]
            reranked_results.append({
                "docs": top_k_docs,
                "is_unanswerable": len(top_k_docs) == 0
            })
    else:
        logger.info("[Stage 3] Starting Cross-Encoder Reranking...")
        t_s3 = time.perf_counter()

        from src.reranker.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranked_results = reranker.rerank_batch(queries, all_docs)

        logger.info("[Stage 3] Reranking done in %.2fs.", time.perf_counter() - t_s3)

    # ============================================================
    # Stage 4: Batch LLM Generation
    # ============================================================
    logger.info("[Stage 4] Starting Batch LLM Generation for %d queries...", len(queries))
    t_s4 = time.perf_counter()

    answers = generator.generate_batch(queries, reranked_results)

    logger.info("[Stage 4] Generation done in %.2fs.", time.perf_counter() - t_s4)

    # ============================================================
    # Stage 5: Write Output
    # ============================================================
    logger.info("[Stage 5] Writing results to: %s", output_file)
    t_s5 = time.perf_counter()

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tong ket thoi gian
    total_elapsed = time.perf_counter() - pipeline_start
    avg_latency = total_elapsed / max(len(queries), 1)
    t_rerank = time.perf_counter() - t_s3 - (time.perf_counter() - t_s4 if 't_s4' in dir() else 0)

    # Map per-query retrieval times (Hop 1 + Hop 2 coalesced)
    # per_query_retrieval_times co the co nhieu hon N entries (do Hop 2)
    # Ta phan bo lai: moi query lay retrieve time cua Hop 1 (entry i*1 hoac i*2)
    # Cach don gian nhat: dung avg retrieval time neu khong map duoc chinh xac
    avg_retrieval = sum(per_query_retrieval_times) / max(len(per_query_retrieval_times), 1)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (qid, query, answer) in enumerate(zip(question_ids, queries, answers)):
            record = {
                "question_id": str(qid),
                "question": query,
                "answer": answer,
                "latency_sec": round(avg_latency, 4),
                "retrieval_latency_sec": round(avg_retrieval, 4),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[Stage 5] Wrote %d answers in %.2fs.", len(answers), time.perf_counter() - t_s5)

    # --- Tong ket ---
    throughput = len(queries) / total_elapsed if total_elapsed > 0 else 0
    n_unanswerable = sum(1 for r in reranked_results if r.get("is_unanswerable", False))

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("  Total questions  : %d", len(queries))
    logger.info("  Unanswerable     : %d (%.1f%%)", n_unanswerable, 100 * n_unanswerable / max(len(queries), 1))
    logger.info("  Total time       : %.2fs", total_elapsed)
    logger.info("  Avg retrieval    : %.4fs/query", avg_retrieval)
    logger.info("  Throughput       : %.1f queries/s", throughput)
    logger.info("  Output           : %s", out_path.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
