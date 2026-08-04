import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from src.trag_v2.retriever_v2 import EnterpriseRetrieverV2

load_dotenv()

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """Extract key technical entities from the following document excerpts.
Entities include: ticket IDs (e.g. JIRA-123), PR numbers (e.g. PR #102), branch names, error codes, feature names, project names.
Return ONLY a comma-separated list on a SINGLE LINE. Do NOT explain. If none found, return exactly "NONE".

Documents:
{context}

Entities:"""

_MAX_ENTITY_LEN = 200


def _parse_entities(raw: str) -> str:
    if not raw or not raw.strip():
        return "NONE"

    raw_stripped = raw.strip()
    if raw_stripped.upper() in ("NONE", "N", "NULL", "UNKNOWN"):
        return "NONE"

    try:
        # Try JSON first just in case
        data = json.loads(raw_stripped)
        if isinstance(data, dict):
            entities = data.get("entities", [])
        elif isinstance(data, list):
            entities = data
        else:
            entities = [str(data)]
    except json.JSONDecodeError:
        # Fallback to comma separated list (expected behavior)
        entities = [e.strip() for e in raw_stripped.split(",")]

    if not isinstance(entities, list):
        return "NONE"

    clean_entities = []
    for e in entities:
        e = str(e).strip()
        if e and e.upper() not in ("NONE", "N", "NULL", "UNKNOWN"):
            clean_entities.append(e)

    if not clean_entities:
        return "NONE"

    result = ", ".join(clean_entities)
    return result[:_MAX_ENTITY_LEN]


class CSEPRetrieverV2:
    """
    T-RAG v2 Cross-Source Entity Propagation (CSEP) Retriever
    Tối ưu hóa:
    - Loại bỏ hoàn toàn double-encode
    - Smart Hop 2: Conditional Hop 2 dựa trên độ tương đồng (vector distance) và số active shards.
    """

    def __init__(
        self,
        retriever=None,
        llm_generate_fn=None,
        top_k_retrieve: int = None,
        smart_hop2: bool = True,
        hop1_dist_threshold: float = 0.55,
        csep: bool = True,
    ):
        self.top_k_retrieve = top_k_retrieve or int(os.environ.get("RAG_TOP_K_RETRIEVE", 20))
        self.retrieve_workers = int(os.environ.get("RAG_RETRIEVE_WORKERS", 8))

        self.smart_hop2 = smart_hop2
        self.hop1_dist_threshold = hop1_dist_threshold
        self.csep = csep

        logger.info(
            f"[CSEP v2] Init: smart_hop2={self.smart_hop2}, threshold={self.hop1_dist_threshold}, csep={self.csep}"
        )

        if retriever is None:
            self.retriever = EnterpriseRetrieverV2()
        else:
            self.retriever = retriever

        self.llm_generate_fn = llm_generate_fn

    def _extract_entities_batch(self, anchor_docs_per_query: List[List[Dict]]) -> List[str]:
        if self.llm_generate_fn is None:
            return ["NONE"] * len(anchor_docs_per_query)

        prompts = []
        for docs in anchor_docs_per_query:
            context = "\n---\n".join(d.get("content", "")[:300] for d in docs[:3])
            prompts.append(ENTITY_EXTRACTION_PROMPT.format(context=context))

        logger.info(f"[CSEP v2] Entity extraction: calling LLM with {len(prompts)} prompts.")
        raw_responses = self.llm_generate_fn(prompts)

        entities = []
        for resp in raw_responses:
            entities.append(_parse_entities(resp))
        return entities

    def retrieve_batch(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        n = len(queries)
        logger.info(f"[CSEP v2] INPUT: {n} queries.")

        t0 = time.perf_counter()
        hop1_results: List[List[Dict]] = [None] * n
        hop1_source_probs: List[Dict[str, float]] = [None] * n
        hop1_embs: List[np.ndarray] = [None] * n

        def process_query_hop1(idx: int, query: str):
            docs, source_probs, emb = self.retriever.retrieve(query, top_k=self.top_k_retrieve)
            return idx, docs, source_probs, emb

        with ThreadPoolExecutor(max_workers=self.retrieve_workers) as executor:
            futures = [executor.submit(process_query_hop1, idx, q) for idx, q in enumerate(queries)]
            for fut in futures:
                idx, docs, source_probs, emb = fut.result()
                hop1_results[idx] = docs
                hop1_source_probs[idx] = source_probs
                hop1_embs[idx] = emb

        elapsed_hop1 = time.perf_counter() - t0
        logger.info(f"[CSEP v2] Hop 1 done in {elapsed_hop1:.2f}s.")

        if not self.csep or self.llm_generate_fn is None:
            logger.info("[CSEP v2] CSEP disabled or no LLM. Skipping Hop 2 entirely.")
            return hop1_results

        csep_candidate_flags = [True] * n
        for i in range(n):
            if self.smart_hop2:
                probs_arr = np.array(list(hop1_source_probs[i].values()))
                if self.retriever.adaptive_tau:
                    tau_eff = self.retriever.compute_adaptive_tau(probs_arr)
                else:
                    tau_eff = self.retriever.tau_base

                active_count = sum(1 for p in hop1_source_probs[i].values() if p >= tau_eff)
                if active_count < 2:
                    csep_candidate_flags[i] = False
                    continue

                docs = hop1_results[i]
                if docs:
                    top_distance = docs[0].get("vector_distance", 1.0)
                    if top_distance < self.hop1_dist_threshold:
                        csep_candidate_flags[i] = False
                        continue

        csep_indices = [i for i, flag in enumerate(csep_candidate_flags) if flag]
        n_csep = len(csep_indices)

        if n_csep == 0:
            logger.info("[CSEP v2] Smart Hop 2: 0 queries needed Hop 2. Skipping LLM entity extraction.")
            return hop1_results

        logger.info(f"[CSEP v2] Smart Hop 2: Only {n_csep}/{n} queries require Hop 2.")

        anchor_docs_for_csep = [hop1_results[i] for i in csep_indices]
        entities_for_csep = self._extract_entities_batch(anchor_docs_for_csep)

        entities_per_query: List[Optional[str]] = [None] * n
        for rank, orig_idx in enumerate(csep_indices):
            entities_per_query[orig_idx] = entities_for_csep[rank]

        t2 = time.perf_counter()
        hop2_results: List[Optional[List[Dict]]] = [None] * n
        hop2_queries = {}

        for i in csep_indices:
            entity_str = entities_per_query[i] or "NONE"
            if entity_str == "NONE" or len(entity_str.strip()) < 3:
                continue

            augmented_query = queries[i] + " " + entity_str
            hop2_queries[i] = augmented_query

        if hop2_queries:
            active_indices = list(hop2_queries.keys())
            active_aug_queries = [hop2_queries[idx] for idx in active_indices]

            with ThreadPoolExecutor(max_workers=self.retrieve_workers) as executor:
                active_hop2_results = list(
                    executor.map(
                        lambda q: self.retriever.retrieve(q, top_k=self.top_k_retrieve)[0],
                        active_aug_queries,
                    )
                )

            for idx, res in zip(active_indices, active_hop2_results):
                hop2_results[idx] = res

        elapsed_hop2 = time.perf_counter() - t2
        logger.info(f"[CSEP v2] Hop 2 completed in {elapsed_hop2:.2f}s for {len(hop2_queries)} executed queries.")

        final_results: List[List[Dict]] = []
        for i in range(n):
            merged = list(hop1_results[i])
            if hop2_results[i]:
                seen_ids = {d.get("doc_id") for d in merged}
                for doc in hop2_results[i]:
                    if doc.get("doc_id") not in seen_ids:
                        merged.append(doc)
                        seen_ids.add(doc.get("doc_id"))

            merged.sort(key=lambda x: x.get("sw_rrf_score", 0.0), reverse=True)
            final_results.append(merged[: self.top_k_retrieve])

        return final_results
