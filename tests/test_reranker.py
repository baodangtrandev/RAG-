"""
Unit Tests for CrossEncoderReranker
Dùng mock model để tránh load model thật (tốn thời gian).
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_reranker():
    """Khởi tạo Reranker với model được mock."""
    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        
        from src.reranker.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="mock-model", threshold=0.5)
        reranker.model = mock_model
        return reranker, mock_model


def test_rerank_batch_filters_low_score_docs(mock_reranker):
    """Tài liệu có score thấp phải bị loại bỏ."""
    reranker, mock_model = mock_reranker

    queries = ["What is the OOM error fix?"]
    docs_per_query = [[
        {"content": "This doc is highly relevant to OOM error fix.", "doc_id": "A"},
        {"content": "Paris is a city in France.", "doc_id": "B"},  # irrelevant
    ]]

    # Mock: doc A score=0.9 (giữ), doc B score=0.1 (bị loại do < threshold=0.5)
    mock_model.predict.return_value = [0.9, 0.1]

    results = reranker.rerank_batch(queries, docs_per_query)

    assert len(results) == 1
    assert not results[0]["is_unanswerable"]
    assert len(results[0]["docs"]) == 1
    assert results[0]["docs"][0]["doc_id"] == "A"
    assert results[0]["docs"][0]["rerank_score"] == pytest.approx(0.9)


def test_rerank_batch_sorted_descending(mock_reranker):
    """Docs phải được sắp xếp theo score giảm dần."""
    reranker, mock_model = mock_reranker
    reranker.threshold = 0.0  # Không lọc, chỉ sort

    queries = ["test query"]
    docs_per_query = [[
        {"content": "doc C", "doc_id": "C"},
        {"content": "doc A", "doc_id": "A"},
        {"content": "doc B", "doc_id": "B"},
    ]]
    mock_model.predict.return_value = [0.3, 0.9, 0.6]

    results = reranker.rerank_batch(queries, docs_per_query)
    doc_ids = [d["doc_id"] for d in results[0]["docs"]]
    assert doc_ids == ["A", "B", "C"], f"Expected ['A','B','C'], got {doc_ids}"


def test_rerank_batch_unanswerable_when_all_below_threshold(mock_reranker):
    """Nếu tất cả docs bị lọc → is_unanswerable=True."""
    reranker, mock_model = mock_reranker
    reranker.threshold = 0.9  # Threshold rất cao

    queries = ["hard question"]
    docs_per_query = [[
        {"content": "doc 1", "doc_id": "X"},
        {"content": "doc 2", "doc_id": "Y"},
    ]]
    mock_model.predict.return_value = [0.2, 0.3]  # Cả hai đều thấp hơn 0.9

    results = reranker.rerank_batch(queries, docs_per_query)
    assert results[0]["is_unanswerable"] is True
    assert results[0]["docs"] == []


def test_rerank_batch_empty_input(mock_reranker):
    """Input rỗng phải trả về list rỗng, không crash."""
    reranker, mock_model = mock_reranker

    results = reranker.rerank_batch([], [])
    assert results == []


def test_rerank_batch_preserves_doc_fields(mock_reranker):
    """Các field gốc của doc phải được giữ nguyên sau rerank."""
    reranker, mock_model = mock_reranker
    reranker.threshold = 0.0

    queries = ["query"]
    docs_per_query = [[
        {
            "content": "relevant content",
            "doc_id": "D1",
            "source": "confluence",
            "title": "My Title",
            "sw_rrf_score": 0.014,
        }
    ]]
    mock_model.predict.return_value = [0.8]

    results = reranker.rerank_batch(queries, docs_per_query)
    doc = results[0]["docs"][0]

    assert doc["doc_id"] == "D1"
    assert doc["source"] == "confluence"
    assert doc["title"] == "My Title"
    assert doc["sw_rrf_score"] == pytest.approx(0.014)
    assert "rerank_score" in doc  # Field mới được thêm vào
