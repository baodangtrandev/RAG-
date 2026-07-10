import argparse
import logging
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("Missing required package. Please run: pip install langchain-text-splitters")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Chunking Configuration
# -----------------------------------------------------------------------------

# Markdown Header configuration for structural splitting
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False  # Retain headers in the resulting chunks for context
)

# Recursive fallback configuration for lengthy paragraphs
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    # Priority: Double newline -> List items -> Single newline -> Sentence end -> Space
    separators=["\n\n", "\n- ", "\n* ", "\n", ". ", " ", ""],
)


def preprocess_text(text: str, source_type: str) -> str:
    """
    Preprocess raw text (pseudo-markdown) into strict Markdown format.
    This ensures that MarkdownHeaderTextSplitter can accurately parse the document structure.
    Uses a source-agnostic scanning strategy (excluding continuous chat logs like Slack).
    """
    if not text or not isinstance(text, str):
        return str(text) if text else ""

    source_type = str(source_type).lower().strip()

    # Slack is a continuous chat log, natural line breaks are sufficient.
    if source_type != "slack":
        # Convert standalone keyword fields (e.g., "Summary:", "description:") to Markdown headers
        text = re.sub(r"(?m)^([A-Za-z0-9][A-Za-z0-9_ \-]{2,80}):\s*$", r"## \1:", text)

        # Convert standard email headers (From:, To:, Subject:, Date:, Cc:)
        text = re.sub(r"(?m)^(From|To|Subject|Date|Cc):\s*", r"## \1: ", text)

        # Convert short, capitalized standalone lines (without ending punctuation) into headers
        text = re.sub(r"(?m)^([A-Z][a-zA-Z0-9 &()_\-]{2,80})$", r"## \1", text)

    return text


def process_row(row_dict: dict) -> list[dict]:
    """
    Standalone worker function to process a single document.
    Designed for parallel execution via ProcessPoolExecutor.
    """
    content = str(row_dict.get("content", "")) if pd.notna(row_dict.get("content")) else ""
    title = str(row_dict.get("title", "")) if "title" in row_dict and pd.notna(row_dict.get("title")) else ""
    original_doc_id = str(row_dict.get("doc_id"))
    source_type = str(row_dict.get("source_type", ""))

    if not content.strip():
        return []

    # Step 1: Preprocess and normalize pseudo-markdown headers
    content = preprocess_text(content, source_type)

    # Prepend the original title as Header 1 to anchor the document context
    full_content = f"# Title: {title}\n\n{content}" if title.strip() else content

    try:
        # Step 2: Attempt structural Markdown splitting
        md_docs = MARKDOWN_SPLITTER.split_text(full_content)
        # Apply recursive splitting for chunks that still exceed the token limit
        final_chunks = TEXT_SPLITTER.split_documents(md_docs)
    except Exception as e:
        # Fallback to pure recursive splitting if Markdown parsing encounters unexpected syntax errors
        logger.warning(
            f"Markdown splitting failed for doc {original_doc_id}: {e}. Falling back to recursive splitting."
        )
        final_chunks = TEXT_SPLITTER.create_documents([full_content])

    results = []
    for c_idx, doc in enumerate(final_chunks):
        new_row = row_dict.copy()

        # Extract header lineage (e.g., "Header 1: Title > Header 2: Description")
        header_vals = [str(v) for k, v in doc.metadata.items() if k.startswith("Header")]
        header_context = " > ".join(header_vals)

        chunk_text = doc.page_content.strip()

        # Step 3: Context Injection for Orphaned Chunks
        # If the recursive splitter breaks a section in half, inject the header lineage
        # at the top of the orphaned chunk to preserve semantic context for embedding models.
        if header_context and not chunk_text.lstrip().startswith("#"):
            chunk_text = f"[{header_context}]\n{chunk_text}"

        new_row["content"] = chunk_text
        new_row["chunk_id"] = c_idx
        new_row["original_doc_id"] = original_doc_id
        new_row["doc_id"] = f"{original_doc_id}_chunk{c_idx}"

        results.append(new_row)

    return results


def main(input_file: str, output_file: str, max_workers: int):
    logger.info(f"Loading data from {input_file}...")

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    df = pd.read_parquet(input_file)

    # Normalize numpy array objects in pandas columns if present
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, "item") else x)

    logger.info(f"Starting Markdown Chunking Pipeline for {len(df)} documents using {max_workers} CPU cores.")

    rows = df.to_dict("records")
    all_chunks = []

    # Execute chunking in parallel to maximize throughput
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use a large chunksize to minimize inter-process communication overhead
        results = list(tqdm(executor.map(process_row, rows, chunksize=100), total=len(rows), desc="Chunking Progress"))

    for res in results:
        all_chunks.extend(res)

    chunked_df = pd.DataFrame(all_chunks)

    logger.info("Pipeline Execution Completed.")
    logger.info(f"Original documents: {len(df)}")
    logger.info(f"Generated chunks: {len(chunked_df)}")
    logger.info(f"Average chunks per document: {len(chunked_df)/len(df) if len(df) > 0 else 0:.2f}")

    logger.info(f"Saving output to {output_file}...")
    chunked_df.to_parquet(output_file)
    logger.info("Save successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise Markdown Chunking Pipeline")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input parquet file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output parquet file")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.workers)
