import os
import sys

# Append project root to path so we can import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.markdown_chunker import preprocess_text, process_row  # noqa: E402


def test_preprocess_text_standard_markdown():
    """Test if standard markdown remains largely unaffected."""
    raw_text = "# Header 1\nSome content."
    processed = preprocess_text(raw_text, "general")
    assert "# Header 1" in processed
    assert "Some content" in processed


def test_preprocess_text_pseudo_headers():
    """Test if pseudo headers (e.g. 'Summary:') are correctly converted to Markdown headers."""
    raw_text = "Summary:\nThis is a summary.\n\nDescription:\nThis is a description."
    processed = preprocess_text(raw_text, "jira")
    assert "## Summary:" in processed
    assert "## Description:" in processed


def test_preprocess_text_email_headers():
    """Test if email standard headers are converted to markdown."""
    raw_text = "From: John Doe\nTo: Jane Doe\nSubject: Project Update"
    processed = preprocess_text(raw_text, "gmail")
    assert "## From:" in processed
    assert "## To:" in processed
    assert "## Subject:" in processed


def test_preprocess_text_slack_exclusion():
    """Test that Slack logs are not aggressively converted into headers."""
    raw_text = "Alice: Hey, how are you?\nBob: I'm good."
    processed = preprocess_text(raw_text, "slack")
    # Should not convert Alice: to ## Alice: because source is slack
    assert "## Alice:" not in processed
    assert "Alice: Hey" in processed


def test_process_row_basic_chunking():
    """Test if a single row is chunked correctly and context is injected."""
    row_dict = {
        "doc_id": "test_doc_001",
        "title": "My Test Doc",
        "source_type": "jira",
        "content": "Description:\nThis is a short description.",
    }

    results = process_row(row_dict)
    assert len(results) > 0
    # Check if title was prepended as a header
    assert "# Title: My Test Doc" in results[0]["content"]
    # Check if doc_id was updated
    assert results[0]["doc_id"] == "test_doc_001_chunk0"
    assert results[0]["original_doc_id"] == "test_doc_001"
