"""
Cross-Source Entity Propagation (CSEP) Retriever
=================================================
Thuat toan Multi-hop Retrieval de giai quyet cac truy van da nguon phuc tap.
Vi du: "Feature ABC tren Slack da duoc merge tren Github chua?"

TOGGLE qua bien moi truong ENABLE_CSEP_FOR_ALL:
    - "True"  => CSEP kich hoat cho TOAN BO cau hoi (mac dinh)
    - "False" => Chi kich hoat khi PSR Router tra ve >= 2 nguon co
                 P(source|query) >= tau (tuc la cau hoi thuc su da nguon)

QUAN TRONG VE PERFORMANCE (Stage-based Batching):
    Entity Extraction PHAI chay theo Batch Stage. Khong duoc goi LLM
    tuan tu trong vong lap for-loop vi se pha vo uu the cua vLLM PagedAttention.

    Dung chay luong:
        Hop 1 retrieval (tat ca N queries)
        -> Batch entity extraction (1 lan goi LLM voi N prompts)
        -> Hop 2 retrieval (tat ca N queries voi augmented query)
        -> Gop ket qua

Config doc tu .env:
    ENABLE_CSEP_FOR_ALL  -- "True"/"False"
    RAG_TAU, RAG_GAMMA, RAG_K_RRF, RAG_TOP_K_RETRIEVE  -- retrieval params
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    logger.warning("[CSEP] Invalid bool value for %s='%s', using default %s", key, val, default)
    return default


def _load_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("[CSEP] Invalid int value for %s, using default %s", key, default)
        return default


ENTITY_EXTRACTION_PROMPT = """Extract key technical entities from the following document excerpts.
Entities include: ticket IDs (e.g. JIRA-123), PR numbers (e.g. PR #102), branch names, error codes, feature names, project names.
Return ONLY a comma-separated list on a SINGLE LINE. Do NOT explain. If none found, return exactly "NONE".

Documents:
{context}

Entities:"""

import json

# Max ky tu cho entity string (chan babbling)
_MAX_ENTITY_LEN = 200


def _parse_entities(raw: str) -> str:
    """
    Parse chuoi JSON do vLLM sinh ra (guided_json).
    Tra ve chuoi entity hoac "NONE".
    """
    if not raw or not raw.strip():
        return "NONE"

    try:
        data = json.loads(raw.strip())
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            return "NONE"

        # Filter and clean
        clean_entities = []
        for e in entities:
            e = str(e).strip()
            if e and e.upper() not in ("NONE", "N", "NULL", "UNKNOWN"):
                clean_entities.append(e)

        if not clean_entities:
            return "NONE"

        result = ", ".join(clean_entities)
        return result[:_MAX_ENTITY_LEN]
    except json.JSONDecodeError:
        # Fallback: neu LLM sinh text cat ngang hong the parse, tot nhat tra ve NONE
        # de khong bi noi chuoi raw JSON vao query gay nhieu vector space.
        return "NONE"


class CSEPRetriever:
    """
    Wraps EnterpriseRetriever voi kha nang Multi-hop Retrieval.

    Su dung pattern Batch Stage: gom tat ca queries lai truoc khi
    goi LLM mot lan duy nhat cho entity extraction.
    """

    def __init__(self, retriever=None, llm_generate_fn=None, top_k_retrieve: int = None):
        """
        Args:
            retriever:       Instance cua EnterpriseRetriever (hoac None de tu tao).
            llm_generate_fn: Ham (prompts: List[str]) -> List[str] de goi LLM.
                             Neu None, entity extraction se bi skip (chi dung Hop 1).
            top_k_retrieve:  So docs moi query lay ra moi hop.
        """
        from dotenv import load_dotenv

        load_dotenv()

        self.enable_csep_for_all = _load_env_bool("ENABLE_CSEP_FOR_ALL", True)
        self.top_k_retrieve = top_k_retrieve if top_k_retrieve is not None else _load_env_int("RAG_TOP_K_RETRIEVE", 20)
        self.retrieve_workers = _load_env_int("RAG_RETRIEVE_WORKERS", 8)

        logger.info(
            "[CSEP] Init: enable_csep_for_all=%s, top_k_retrieve=%d, retrieve_workers=%d",
            self.enable_csep_for_all,
            self.top_k_retrieve,
            self.retrieve_workers,
        )

        if retriever is None:
            from src.retrieval.retriever import EnterpriseRetriever

            self.retriever = EnterpriseRetriever()
        else:
            self.retriever = retriever

        # Ham goi LLM de extract entities (co the None neu khong co LLM)
        self.llm_generate_fn = llm_generate_fn

    def _should_run_csep(self, source_probs: Dict[str, float]) -> bool:
        """
        Quyet dinh co nen chay CSEP cho mot query cu the khong.
        Neu ENABLE_CSEP_FOR_ALL=True -> luon chay.
        Neu ENABLE_CSEP_FOR_ALL=False -> chi chay khi co >= 2 nguon dang ke.
        """
        if self.enable_csep_for_all:
            return True
        # Dem so nguon co xac suat >= tau (lay tu router)
        tau = self.retriever.tau
        active_count = sum(1 for p in source_probs.values() if p >= tau)
        return active_count >= 2

    def _extract_entities_batch(self, anchor_docs_per_query: List[List[Dict]]) -> List[str]:
        """
        Trich xuat entities tu anchor docs bang LLM batch call.
        Tra ve list N string (co the la "NONE" neu khong tim thay gi).
        """
        if self.llm_generate_fn is None:
            logger.warning("[CSEP] No LLM function provided, skipping entity extraction.")
            return ["NONE"] * len(anchor_docs_per_query)

        prompts = []
        for docs in anchor_docs_per_query:
            context = "\n---\n".join(d.get("content", "")[:300] for d in docs[:3])
            prompts.append(ENTITY_EXTRACTION_PROMPT.format(context=context))

        logger.info("[CSEP] Entity extraction: calling LLM with %d prompts.", len(prompts))
        t0 = time.perf_counter()
        raw_responses = self.llm_generate_fn(prompts)
        elapsed = time.perf_counter() - t0
        logger.info("[CSEP] Entity extraction done in %.2fs.", elapsed)

        entities = []
        for resp in raw_responses:
            parsed = _parse_entities(resp)
            entities.append(parsed)
            logger.debug("[CSEP] Raw: %s -> Parsed: %s", repr(resp[:120]), parsed)

        n_none = sum(1 for e in entities if e == "NONE")
        logger.info("[CSEP] Entity extraction results: %d NONE / %d total.", n_none, len(entities))
        return entities

    def retrieve_batch(
        self,
        queries: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieval chinh — chay Hop 1 (va Hop 2 neu CSEP kich hoat).

        Args:
            queries: N cau hoi.

        Returns:
            List[List[dict]] — N danh sach docs, moi list la ket qua
            tong hop cua Hop 1 (va Hop 2 neu co CSEP).
        """
        n = len(queries)
        logger.info("[CSEP] INPUT: %d queries. CSEP_FOR_ALL=%s", n, self.enable_csep_for_all)

        t0 = time.perf_counter()
        hop1_results: List[List[Dict]] = [None] * n
        hop1_source_probs: List[Dict[str, float]] = [None] * n

        def process_query_hop1(idx: int, query: str):
            docs = self.retriever.retrieve(query, top_k=self.top_k_retrieve)
            emb = self.retriever.router.encoder.encode([query], normalize_embeddings=True)
            probs = self.retriever.router.clf.predict_proba(emb)[0]
            source_probs = {self.retriever.router.classes[j]: float(probs[j]) for j in range(len(probs))}
            return idx, docs, source_probs

        with ThreadPoolExecutor(max_workers=self.retrieve_workers) as executor:
            futures = [executor.submit(process_query_hop1, idx, q) for idx, q in enumerate(queries)]
            for fut in futures:
                idx, docs, source_probs = fut.result()
                hop1_results[idx] = docs
                hop1_source_probs[idx] = source_probs

        elapsed_hop1 = time.perf_counter() - t0
        logger.info("[CSEP] Hop 1 done: %d queries in %.2fs.", n, elapsed_hop1)

        csep_flags = [self._should_run_csep(sp) for sp in hop1_source_probs]
        n_csep = sum(csep_flags)

        if n_csep == 0 or self.llm_generate_fn is None:
            logger.info("[CSEP] CSEP skipped for all queries (flag=False or no LLM).")
            logger.info("[CSEP] OUTPUT: %d queries, hop1 only.", n)
            return hop1_results

        logger.info("[CSEP] CSEP will run for %d/%d queries.", n_csep, n)

        csep_indices = [i for i, flag in enumerate(csep_flags) if flag]
        anchor_docs_for_csep = [hop1_results[i] for i in csep_indices]

        entities_for_csep = self._extract_entities_batch(anchor_docs_for_csep)

        entities_per_query: List[Optional[str]] = [None] * n
        for rank, orig_idx in enumerate(csep_indices):
            entities_per_query[orig_idx] = entities_for_csep[rank]

        t2 = time.perf_counter()
        hop2_results: List[Optional[List[Dict]]] = [None] * n

        n_hop2_run = 0
        n_hop2_skip = 0
        hop2_queries = {}
        for i in csep_indices:
            entity_str = entities_per_query[i] or "NONE"
            # Skip Hop 2 khi entity la NONE hoac qua ngan (< 3 ky tu)
            if entity_str == "NONE" or len(entity_str.strip()) < 3:
                logger.debug("[CSEP] Query[%d]: No valid entities, skip Hop 2.", i)
                n_hop2_skip += 1
                continue

            augmented_query = queries[i] + " " + entity_str
            logger.debug(
                "[CSEP] Query[%d]: augmented_query (first 120 chars): %s",
                i,
                augmented_query[:120],
            )
            hop2_queries[i] = augmented_query
            n_hop2_run += 1

        if hop2_queries:
            active_indices = list(hop2_queries.keys())
            active_aug_queries = [hop2_queries[idx] for idx in active_indices]

            with ThreadPoolExecutor(max_workers=self.retrieve_workers) as executor:
                active_hop2_results = list(
                    executor.map(lambda q: self.retriever.retrieve(q, top_k=self.top_k_retrieve), active_aug_queries)
                )

            for idx, res in zip(active_indices, active_hop2_results):
                hop2_results[idx] = res

        logger.info("[CSEP] Hop 2 summary: %d executed, %d skipped (no entities).", n_hop2_run, n_hop2_skip)

        elapsed_hop2 = time.perf_counter() - t2
        logger.info("[CSEP] Hop 2 done: %d queries in %.2fs.", n_csep, elapsed_hop2)

        # ============================================================
        # Sub-stage D: Merge Hop 1 + Hop 2 (dedup by doc_id)
        # ============================================================
        final_results: List[List[Dict]] = []

        for i in range(n):
            merged = list(hop1_results[i])  # copy
            if hop2_results[i]:
                seen_ids = {d.get("doc_id") for d in merged}
                for doc in hop2_results[i]:
                    if doc.get("doc_id") not in seen_ids:
                        merged.append(doc)
                        seen_ids.add(doc.get("doc_id"))

            # Re-sort by sw_rrf_score sau khi merge
            merged.sort(key=lambda x: x.get("sw_rrf_score", 0.0), reverse=True)
            final_results.append(merged[: self.top_k_retrieve])

            logger.debug(
                "[CSEP] Query[%d]: hop1=%d, hop2=%d, merged=%d docs.",
                i,
                len(hop1_results[i]),
                len(hop2_results[i]) if hop2_results[i] else 0,
                len(final_results[i]),
            )

        logger.info("[CSEP] OUTPUT: %d queries, CSEP applied to %d.", n, n_csep)
        return final_results
