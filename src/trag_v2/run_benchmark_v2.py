import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from dotenv import load_dotenv
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.generator import VLLMGenerator
from src.reranker.reranker import CrossEncoderReranker
from src.trag_v2.csep_retriever_v2 import CSEPRetrieverV2
from src.trag_v2.retriever_v2 import EnterpriseRetrieverV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trag_v2.benchmark")

app = typer.Typer(
    name="trag-v2-benchmark",
    help="T-RAG v2 Pipeline Benchmark Runner",
    add_completion=False,
)


@app.command()
def run(
    questions_file: str = typer.Option(
        "data/EnterpriseRAG-Bench/data/questions/test.parquet",
        "--questions",
        "-q",
        help="Path tới file chứa các câu hỏi (parquet hoặc jsonl).",
    ),
    output_file: str = typer.Option(
        "results_v6/trag_v2_standard.jsonl",
        "--output",
        "-o",
        help="File kết quả đầu ra (JSONL).",
    ),
    tau_base: float = typer.Option(
        0.15,
        "--tau-base",
        help="Ngưỡng router cơ sở (tau_base).",
    ),
    tau_alpha: float = typer.Option(
        0.08,
        "--tau-alpha",
        help="Hệ số tự thích ứng (alpha) cho adaptive tau.",
    ),
    gamma: float = typer.Option(
        0.5,
        "--gamma",
        help="Hệ số phạt nguồn (gamma).",
    ),
    top_k_retrieve: int = typer.Option(
        20,
        "--top-k-retrieve",
        help="Số lượng docs truy xuất tối đa trước reranker.",
    ),
    top_k_final: int = typer.Option(
        7,
        "--top-k-final",
        help="Số lượng docs đưa vào LLM sau reranker.",
    ),
    smart_hop2: bool = typer.Option(
        True,
        "--smart-hop2/--no-smart-hop2",
        help="Bật/Tắt tính năng Smart Hop 2 (Conditional Hop 2).",
    ),
    adaptive_tau: bool = typer.Option(
        True,
        "--adaptive-tau/--no-adaptive-tau",
        help="Bật/Tắt tính năng Adaptive Tau.",
    ),
    csep: bool = typer.Option(
        True,
        "--csep/--no-csep",
        help="Bật/Tắt hoàn toàn module CSEP.",
    ),
    dense_weight: float = typer.Option(
        0.5,
        "--dense-weight",
        help="Trọng số tìm kiếm Dense (vector).",
    ),
    sparse_weight: float = typer.Option(
        0.5,
        "--sparse-weight",
        help="Trọng số tìm kiếm Sparse (BM25).",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Giới hạn số câu hỏi để test nhanh.",
    ),
):
    """
    Chạy T-RAG v2 Pipeline trên tập dữ liệu benchmark.
    """
    logger.info("=" * 60)
    logger.info("T-RAG v2 BENCHMARK CONFIG")
    logger.info(f"  Questions File  : {questions_file}")
    logger.info(f"  Output File     : {output_file}")
    logger.info(f"  Tau Base        : {tau_base}")
    logger.info(f"  Tau Alpha       : {tau_alpha}")
    logger.info(f"  Gamma           : {gamma}")
    logger.info(f"  Top K Retrieve  : {top_k_retrieve}")
    logger.info(f"  Top K Final     : {top_k_final}")
    logger.info(f"  Smart Hop 2     : {smart_hop2}")
    logger.info(f"  Adaptive Tau    : {adaptive_tau}")
    logger.info(f"  CSEP Enabled    : {csep}")
    logger.info(f"  Dense Weight    : {dense_weight}")
    logger.info(f"  Sparse Weight   : {sparse_weight}")
    logger.info("=" * 60)

    logger.info("[Stage 0] Initializing vLLM Generator...")
    generator = VLLMGenerator(top_k_final=top_k_final)

    pipeline_start = time.perf_counter()

    logger.info(f"[Stage 1] Loading questions from: {questions_file}")
    qpath = Path(questions_file)
    if not qpath.exists():
        logger.error(f"Questions file not found: {questions_file}")
        raise typer.Exit(code=1)

    if qpath.suffix == ".parquet":
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
        logger.error(f"Unsupported file format: {qpath.suffix}")
        raise typer.Exit(code=1)

    if limit:
        queries = queries[:limit]
        question_ids = question_ids[:limit]

    logger.info(f"[Stage 1] Loaded {len(queries)} questions.")

    logger.info("[Stage 2] Running CSEP Retrieval...")

    retriever = EnterpriseRetrieverV2(
        tau_base=tau_base,
        tau_alpha=tau_alpha,
        adaptive_tau=adaptive_tau,
        gamma=gamma,
        k_rrf=60,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    def csep_llm_fn(prompts):
        schema = {
            "type": "object",
            "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["entities"],
        }
        guided_decoding = GuidedDecodingParams(json=schema)
        entity_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["\n\n", "\nDocuments:", "\nEntities:", "<|im_end|>", "<|endoftext|>", "NONE", ", NONE"],
            guided_decoding=guided_decoding,
        )
        outputs = generator.llm.generate(prompts, entity_params)
        return [o.outputs[0].text.strip() for o in outputs]

    csep_retriever = CSEPRetrieverV2(
        retriever=retriever,
        llm_generate_fn=csep_llm_fn,
        top_k_retrieve=top_k_retrieve,
        smart_hop2=smart_hop2,
        csep=csep,
    )

    t_ret_start = time.perf_counter()
    all_docs = csep_retriever.retrieve_batch(queries)
    batch_retrieval_time = time.perf_counter() - t_ret_start
    avg_retrieval = batch_retrieval_time / max(len(queries), 1)

    logger.info(f"[Stage 2] Retrieval completed in {batch_retrieval_time:.2f}s.")

    logger.info("[Stage 3] Running Cross-Encoder Reranking...")
    t_s3 = time.perf_counter()

    reranker = CrossEncoderReranker()
    reranked_results = reranker.rerank_batch(queries, all_docs)
    logger.info(f"[Stage 3] Reranking completed in {time.perf_counter() - t_s3:.2f}s.")

    logger.info("[Stage 4] Running LLM Generation...")
    t_s4 = time.perf_counter()
    answers = generator.generate_batch(queries, reranked_results)
    logger.info(f"[Stage 4] Generation completed in {time.perf_counter() - t_s4:.2f}s.")

    logger.info(f"[Stage 5] Saving results to file: {output_file}")
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_elapsed = time.perf_counter() - pipeline_start
    avg_latency = total_elapsed / max(len(queries), 1)

    refused_count = 0
    search_spaces = []

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (qid, query, answer) in enumerate(zip(question_ids, queries, answers)):
            docs_for_query = all_docs[idx]
            search_space = docs_for_query[0].get("search_space_docs", 0) if docs_for_query else 0
            search_spaces.append(search_space)

            if "do not have enough" in answer.lower() or "i don" in answer.lower():
                refused_count += 1

            record = {
                "question_id": str(qid),
                "question": query,
                "answer": answer,
                "latency_sec": round(avg_latency, 4),
                "retrieval_latency_sec": round(avg_retrieval, 4),
                "search_space_docs": int(search_space),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_search_space = sum(search_spaces) / max(len(search_spaces), 1)
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"  Total queries   : {len(queries)}")
    logger.info(f"  Avg Latency     : {avg_latency:.4f}s")
    logger.info(f"  Avg Retrieval   : {avg_retrieval:.4f}s")
    logger.info(f"  Avg Search Space: {avg_search_space:,.0f} docs")
    logger.info(
        f"  Refused Rate    : {refused_count}/{len(queries)} ({100 * refused_count / max(len(queries), 1):.1f}%)"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
