"""
Unit Tests for CSEPRetriever
Mock EnterpriseRetriever va LLM function de test logic CSEP.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_retriever(tau=0.15, classes=None, probs=None):
    """Tao mock EnterpriseRetriever voi router duoc dinh san."""
    if classes is None:
        classes = ["slack", "jira", "github", "confluence"]
    if probs is None:
        probs = [0.8, 0.6, 0.1, 0.05]  # slack + jira cao hon tau

    mock_retriever = MagicMock()
    mock_retriever.tau = tau

    # Setup router mock
    mock_retriever.router.classes = classes
    mock_retriever.router.encoder.encode.return_value = [[0.1] * 1024]
    mock_retriever.router.clf.predict_proba.return_value = [probs]

    # retrieve() mac dinh tra ve 3 docs
    mock_retriever.retrieve.return_value = [
        {"doc_id": "d1", "content": "JIRA-123 is the ticket", "source": "slack", "sw_rrf_score": 0.9},
        {"doc_id": "d2", "content": "Some other content", "source": "slack", "sw_rrf_score": 0.7},
    ]

    return mock_retriever


def _make_csep(enable_for_all: bool, retriever, llm_fn=None, top_k=10):
    """Tao CSEPRetriever ma khong goi __init__ day du (tranh load real model)."""
    import importlib
    import src.retrieval.csep_retriever as m
    importlib.reload(m)
    from src.retrieval.csep_retriever import CSEPRetriever

    csep = CSEPRetriever.__new__(CSEPRetriever)
    csep.enable_csep_for_all = enable_for_all
    csep.top_k_retrieve = top_k
    csep.retriever = retriever
    csep.llm_generate_fn = llm_fn
    return csep


# ======== Tests for _should_run_csep ========

def test_should_run_csep_always_true_when_flag_on():
    retriever = _make_mock_retriever()
    csep = _make_csep(enable_for_all=True, retriever=retriever)
    source_probs = {"slack": 0.01, "jira": 0.01}  # tat ca thap
    assert csep._should_run_csep(source_probs) is True


def test_should_run_csep_false_when_single_source():
    """Khi CSEP_FOR_ALL=False va chi 1 nguon > tau, CSEP khong nen chay."""
    retriever = _make_mock_retriever(tau=0.15)
    csep = _make_csep(enable_for_all=False, retriever=retriever)
    source_probs = {"slack": 0.9, "jira": 0.05, "github": 0.02}
    assert csep._should_run_csep(source_probs) is False


def test_should_run_csep_true_when_multiple_sources():
    """Khi CSEP_FOR_ALL=False nhung co >= 2 nguon > tau, CSEP nen chay."""
    retriever = _make_mock_retriever(tau=0.15)
    csep = _make_csep(enable_for_all=False, retriever=retriever)
    source_probs = {"slack": 0.9, "jira": 0.7, "github": 0.02}
    assert csep._should_run_csep(source_probs) is True


# ======== Tests for _extract_entities_batch ========

def test_extract_entities_batch_calls_llm_with_correct_count():
    """LLM duoc goi voi dung so prompts = so query."""
    mock_llm = MagicMock(return_value=["JIRA-123", "PR-456"])
    retriever = _make_mock_retriever()
    csep = _make_csep(enable_for_all=True, retriever=retriever, llm_fn=mock_llm)

    anchor_docs = [
        [{"content": "Slack says JIRA-123 is blocked"}],
        [{"content": "PR-456 was merged"}],
    ]
    result = csep._extract_entities_batch(anchor_docs)

    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[0][0]
    assert len(call_args) == 2  # 2 prompts
    assert result == ["JIRA-123", "PR-456"]


def test_extract_entities_batch_no_llm_returns_none():
    """Neu khong co LLM, phai tra ve list NONE, khong crash."""
    retriever = _make_mock_retriever()
    csep = _make_csep(enable_for_all=True, retriever=retriever, llm_fn=None)

    result = csep._extract_entities_batch([[{"content": "some doc"}]])
    assert result == ["NONE"]


# ======== Tests for retrieve_batch logic ========

def test_retrieve_batch_csep_off_returns_hop1_only():
    """Khi CSEP_FOR_ALL=False va chi 1 nguon active, chi Hop 1 duoc chay."""
    # Chỉ 1 nguon vuot qua tau -> CSEP khong kich hoat
    retriever = _make_mock_retriever(
        probs=[0.9, 0.05, 0.01, 0.01]  # chi slack > 0.15
    )
    mock_llm = MagicMock(return_value=["NONE"])
    csep = _make_csep(enable_for_all=False, retriever=retriever, llm_fn=mock_llm)

    results = csep.retrieve_batch(["What happened with the Slack thread?"])

    assert len(results) == 1
    # LLM khong duoc goi vi CSEP khong kich hoat
    mock_llm.assert_not_called()


def test_retrieve_batch_csep_on_augments_query():
    """Khi CSEP bat, Hop 2 phai duoc goi voi augmented query chua entity."""
    retriever = _make_mock_retriever(
        probs=[0.9, 0.8, 0.01, 0.01]  # slack + jira > tau -> CSEP=True
    )
    mock_llm = MagicMock(return_value=["JIRA-999"])
    csep = _make_csep(enable_for_all=True, retriever=retriever, llm_fn=mock_llm)

    csep.retrieve_batch(["Has Slack feature been closed in Jira?"])

    # retrieve() duoc goi >= 2 lan (hop1 + hop2)
    call_count = retriever.retrieve.call_count
    assert call_count >= 2

    # Lan goi thu 2 (hop2) phai chua entity "JIRA-999"
    second_call_query = retriever.retrieve.call_args_list[1][0][0]
    assert "JIRA-999" in second_call_query


def test_retrieve_batch_deduplicates_docs():
    """Doc xuat hien o ca hop1 va hop2 chi co trong merged result mot lan."""
    hop1_docs = [
        {"doc_id": "shared", "content": "shared doc", "source": "slack", "sw_rrf_score": 0.9},
    ]
    hop2_docs = [
        {"doc_id": "shared", "content": "shared doc", "source": "slack", "sw_rrf_score": 0.9},
        {"doc_id": "unique", "content": "unique doc", "source": "jira", "sw_rrf_score": 0.6},
    ]

    retriever = _make_mock_retriever(probs=[0.9, 0.8, 0.01, 0.01])
    retriever.retrieve.side_effect = [hop1_docs, hop2_docs]

    mock_llm = MagicMock(return_value=["JIRA-999"])
    csep = _make_csep(enable_for_all=True, retriever=retriever, llm_fn=mock_llm)

    results = csep.retrieve_batch(["question"])
    doc_ids = [d["doc_id"] for d in results[0]]

    assert doc_ids.count("shared") == 1, "shared doc phai xuat hien duy nhat 1 lan"
    assert "unique" in doc_ids
