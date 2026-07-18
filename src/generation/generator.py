"""
Generator Module (Local vLLM Integration)
==========================================
Su dung vLLM Offline Batching de sinh cau tra loi tu local LLM.
Thiet ke Stage-based: nhan vao list prompts, tra ve list answers —
khong goi LLM theo vong lap tung cau (tranh OOM, tan dung PagedAttention).

Config doc tu .env:
    LOCAL_LLM_MODEL              -- HuggingFace model ID
    VLLM_GPU_MEMORY_UTILIZATION  -- phan tram VRAM danh cho vLLM (mac dinh: 0.8)
    RAG_TOP_K_FINAL              -- so docs toi da dua vao moi prompt

Input:  queries + reranked docs (output tu Reranker)
Output: List[str] answers, moi answer tuong ung voi 1 query
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Prompt template chuan — co the override qua subclass
RAG_SYSTEM_PROMPT = (
    "You are a helpful and precise enterprise assistant. Answer the user question "
    "based on the provided context documents. The documents come from a retrieval system which is imperfect. "
    "Try your best to answer the question using the available information, including drawing reasonable "
    "conclusions and combining facts from multiple documents. "
    "If and only if the context documents contain absolutely no relevant information to the question, "
    "say 'I do not have enough information to answer this question based on the available sources.'"
)

RAG_USER_TEMPLATE = """Answer the question based on the context documents provided below.

<context>
{context}
</context>

Question: {query}

Answer:"""

UNANSWERABLE_RESPONSE = (
    "I do not have enough information to answer this question "
    "based on the available sources."
)


def _load_env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("[Generator] Invalid env value for %s, using default %s", key, default)
        return default


def _load_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("[Generator] Invalid env value for %s, using default %s", key, default)
        return default


class VLLMGenerator:
    """
    LLM Generator su dung vLLM Offline Batching.

    Kien truc: Khoi tao vLLM.LLM mot lan duy nhat (singleton),
    sau do dung llm.generate(prompts) de xu ly batch lon.

    NOTE: vLLM se chiem VLLM_GPU_MEMORY_UTILIZATION% VRAM.
    Neu chay cung Reranker tren cung GPU, can load Reranker TRUOC
    roi moi khoi tao VLLMGenerator de tranh OOM.
    """

    def __init__(
        self,
        model_name: str = None,
        gpu_memory_utilization: float = None,
        top_k_final: int = None,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        self.model_name = (
            model_name
            or os.environ.get("LOCAL_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        )
        self.gpu_memory_utilization = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else _load_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.8)
        )
        self.top_k_final = (
            top_k_final
            if top_k_final is not None
            else _load_env_int("RAG_TOP_K_FINAL", 5)
        )

        logger.info(
            "[Generator] Init vLLM: model='%s', gpu_util=%.2f, top_k_final=%d",
            self.model_name, self.gpu_memory_utilization, self.top_k_final,
        )

        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,   # Thap de dam bao tin cay trong Enterprise RAG
            max_tokens=384,    # Du cho Enterprise answer, tranh babbling
            stop=[
                "\n\nQuestion:",
                "\n\nContext:",
                "<|im_end|>",
                "<|endoftext|>",
            ],
        )

        logger.info("[Generator] vLLM loaded successfully.")

    def build_rag_prompt(
        self,
        query: str,
        docs: List[Dict[str, Any]],
    ) -> str:
        """Xay dung prompt dung Chat Template thay vi noi chuoi tho (de chong babbling)."""
        top_docs = docs[: self.top_k_final]
        context_parts = []
        for d in top_docs:
            txt = d.get("content", "").strip()
            if txt:
                source = d.get("source", "unknown").upper()
                title = d.get("title", "")
                header = f"[{source}]" + (f" {title}" if title else "")
                context_parts.append(f"{header}\n{txt}")

        context = "\n\n".join(context_parts) if context_parts else "No context available."
        user_content = RAG_USER_TEMPLATE.format(context=context, query=query)
        
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        tokenizer = self.llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def generate_batch(
        self,
        queries: List[str],
        reranked_results: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Sinh cau tra loi cho tat ca queries trong mot lan vLLM call.

        Args:
            queries:          N cau hoi goc.
            reranked_results: N entries tu output cua Reranker
                              (moi entry: {"docs": [...], "is_unanswerable": bool}).

        Returns:
            List[str] dai N — moi element la cau tra loi tuong ung.
        """
        assert len(queries) == len(reranked_results), (
            "[Generator] Mismatch: %d queries vs %d reranked results"
            % (len(queries), len(reranked_results))
        )

        n = len(queries)
        logger.info("[Generator] INPUT: %d queries for batch generation.", n)

        # --- Phan loai: unanswerable vs answerable ---
        prompts: List[Optional[str]] = []
        prompt_indices: List[int] = []  # index cua queries can goi LLM

        for i, (query, result) in enumerate(zip(queries, reranked_results)):
            # Support both list of docs (from baselines) and dict with 'docs' key (from T-RAG pipeline)
            if isinstance(result, list):
                docs = result
                is_unanswerable = len(docs) == 0
            else:
                docs = result.get("docs", [])
                is_unanswerable = result.get("is_unanswerable", False)
                
            if is_unanswerable or not docs:
                prompts.append(None)  # Danh dau khong can goi LLM
                logger.debug("[Generator] Query[%d] marked unanswerable, skip LLM.", i)
            else:
                prompt = self.build_rag_prompt(query, docs)
                prompts.append(prompt)
                prompt_indices.append(i)

        n_answerable = len(prompt_indices)
        n_unanswerable = n - n_answerable
        logger.info(
            "[Generator] %d answerable (will call LLM) | %d unanswerable (skip LLM).",
            n_answerable, n_unanswerable,
        )

        # --- Khoi tao answers voi gia tri mac dinh ---
        answers = [UNANSWERABLE_RESPONSE] * n

        if n_answerable == 0:
            logger.warning("[Generator] No answerable queries — skipping LLM call.")
            return answers

        # --- Lay cac prompts can goi LLM ---
        active_prompts = [prompts[i] for i in prompt_indices]

        # Log vi du prompt dau tien de debug
        logger.debug(
            "[Generator] Sample prompt[0] (first %d chars): %s",
            200, active_prompts[0][:200],
        )

        # --- Batch inference (mot lan duy nhat — toan dung H100) ---
        t0 = time.perf_counter()
        vllm_outputs = self.llm.generate(active_prompts, self.sampling_params)
        elapsed = time.perf_counter() - t0

        total_tokens = sum(
            len(o.outputs[0].token_ids) for o in vllm_outputs
        )
        logger.info(
            "[Generator] vLLM inference done: %d prompts | %.2fs | "
            "%d total tokens | %.0f tokens/s",
            n_answerable, elapsed, total_tokens, total_tokens / max(elapsed, 1e-9),
        )

        # --- Map ket qua ve dung index ---
        for rank, original_idx in enumerate(prompt_indices):
            answer_text = vllm_outputs[rank].outputs[0].text.strip()
            answers[original_idx] = answer_text
            logger.debug(
                "[Generator] Query[%d] answer (first 100 chars): %s",
                original_idx, answer_text[:100],
            )

        logger.info("[Generator] OUTPUT: %d answers generated.", n)
        return answers
