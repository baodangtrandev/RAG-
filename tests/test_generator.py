"""
Unit Tests for VLLMGenerator
Dung sys.modules mock de tranh load vLLM that (chua install trong test env).
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Mock vllm truoc khi import generator ----
# vLLM co the chua duoc cai dat trong test environment,
# nen ta inject stub vao sys.modules truoc khi import.
mock_vllm_module = MagicMock()
mock_vllm_module.LLM = MagicMock
mock_vllm_module.SamplingParams = MagicMock
sys.modules.setdefault("vllm", mock_vllm_module)

from src.generation.generator import UNANSWERABLE_RESPONSE, VLLMGenerator  # noqa: E402


def _make_generator(**kwargs) -> VLLMGenerator:
    """Tao VLLMGenerator voi mock llm, khong can GPU."""
    mock_llm = MagicMock()
    default_output = MagicMock()
    default_output.outputs[0].text = "  The fix is to reduce batch size.  "
    default_output.outputs[0].token_ids = [1, 2, 3, 4, 5]
    mock_llm.generate.return_value = [default_output]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: str(messages)
    mock_llm.get_tokenizer.return_value = mock_tokenizer

    gen = VLLMGenerator.__new__(VLLMGenerator)
    gen.model_name = kwargs.get("model_name", "mock-model")
    gen.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.5)
    gen.top_k_final = kwargs.get("top_k_final", 3)
    gen.llm = mock_llm
    gen.sampling_params = MagicMock()
    return gen


# ======== Tests for build_rag_prompt ========


def test_build_rag_prompt_includes_query():
    gen = _make_generator()
    docs = [{"content": "Some content", "source": "confluence", "title": "Doc 1"}]
    prompt = gen.build_rag_prompt("How to fix OOM?", docs)
    assert "How to fix OOM?" in prompt


def test_build_rag_prompt_includes_doc_content():
    gen = _make_generator()
    docs = [{"content": "Reduce batch size to fix OOM.", "source": "github", "title": ""}]
    prompt = gen.build_rag_prompt("How to fix OOM?", docs)
    assert "Reduce batch size to fix OOM." in prompt


def test_build_rag_prompt_respects_top_k_final():
    gen = _make_generator(top_k_final=2)
    docs = [
        {"content": "Doc A", "source": "s1", "title": ""},
        {"content": "Doc B", "source": "s2", "title": ""},
        {"content": "Doc C", "source": "s3", "title": ""},  # phai bi bo qua
    ]
    prompt = gen.build_rag_prompt("query", docs)
    assert "Doc A" in prompt
    assert "Doc B" in prompt
    assert "Doc C" not in prompt


def test_build_rag_prompt_empty_docs():
    gen = _make_generator()
    prompt = gen.build_rag_prompt("query", [])
    assert "No context available" in prompt


def test_build_rag_prompt_source_uppercase():
    gen = _make_generator()
    docs = [{"content": "data", "source": "jira", "title": "Ticket"}]
    prompt = gen.build_rag_prompt("q", docs)
    assert "JIRA" in prompt


# ======== Tests for generate_batch ========


def test_generate_batch_unanswerable_skips_llm():
    gen = _make_generator()
    queries = ["impossible question"]
    reranked = [{"docs": [], "is_unanswerable": True}]

    answers = gen.generate_batch(queries, reranked)

    assert answers[0] == UNANSWERABLE_RESPONSE
    gen.llm.generate.assert_not_called()


def test_generate_batch_calls_llm_for_answerable():
    gen = _make_generator()
    queries = ["How to fix OOM?"]
    reranked = [{"docs": [{"content": "Reduce batch.", "source": "slack", "title": ""}], "is_unanswerable": False}]

    answers = gen.generate_batch(queries, reranked)

    gen.llm.generate.assert_called_once()
    assert answers[0] == "The fix is to reduce batch size."  # stripped


def test_generate_batch_mixed():
    """Mixed: answerable va unanswerable phai duoc xu ly dung."""
    gen = _make_generator()
    mock_output = MagicMock()
    mock_output.outputs[0].text = "Answer for Q1"
    mock_output.outputs[0].token_ids = [1, 2, 3]
    gen.llm.generate.return_value = [mock_output]

    queries = ["Q1 answerable", "Q2 unanswerable"]
    reranked = [
        {"docs": [{"content": "relevant", "source": "jira", "title": ""}], "is_unanswerable": False},
        {"docs": [], "is_unanswerable": True},
    ]

    answers = gen.generate_batch(queries, reranked)

    assert answers[0] == "Answer for Q1"
    assert answers[1] == UNANSWERABLE_RESPONSE
    # LLM chi duoc goi 1 prompt (cho Q1)
    call_args = gen.llm.generate.call_args[0]
    assert len(call_args[0]) == 1


def test_generate_batch_empty_input():
    gen = _make_generator()
    answers = gen.generate_batch([], [])
    assert answers == []
    gen.llm.generate.assert_not_called()


def test_generate_batch_all_unanswerable_skips_llm():
    gen = _make_generator()
    queries = ["q1", "q2"]
    reranked = [
        {"docs": [], "is_unanswerable": True},
        {"docs": [], "is_unanswerable": True},
    ]
    answers = gen.generate_batch(queries, reranked)
    assert all(a == UNANSWERABLE_RESPONSE for a in answers)
    gen.llm.generate.assert_not_called()
