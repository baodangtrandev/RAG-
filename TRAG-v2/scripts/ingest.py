import argparse
import logging
import os
import sys

import lancedb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Append project root to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.ingestion.lance_schema import DocumentSchema
except ImportError:
    raise ImportError("Failed to import DocumentSchema. Ensure src.ingestion.lance_schema exists.")


def main(data_dir: str, file_name: str, db_path: str, batch_size: int, model_name: str, write_batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # [OPTIMIZATION 1]: Force FP16 precision (Bfloat16/Float16).
    # Default FP32 is slow and memory-intensive. FP16 triggers Tensor Cores on modern GPUs (e.g., H100),
    # significantly doubling the speed and halving VRAM consumption.
    logger.info(f"Loading embedding model (FP16 Optimization): {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})

    logger.info(f"Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)

    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Reading chunked data from {file_path}...")
    df = pd.read_parquet(file_path)

    # Normalize numpy array objects from Pandas parquet if present
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, "item") else x)

    # Ensure mandatory columns exist for downstream payload processing
    if "chunk_id" not in df.columns:
        df["chunk_id"] = 0
    if "original_doc_id" not in df.columns:
        df["original_doc_id"] = df["doc_id"]

    source_types = df["source_type"].unique()
    logger.info(f"Found {len(source_types)} unique source types for Sharding: {list(source_types)}")

    # PHYSICAL SHARDING STRATEGY: Process each source type into its own dedicated LanceDB table
    for src in source_types:
        table_name = str(src).lower().strip().replace(" ", "_").replace("-", "_")

        subset = df[df["source_type"] == src]
        total_docs = len(subset)
        logger.info(f"--- Processing source (Table): {table_name} | {total_docs} chunks ---")

        tbl = None
        if table_name in db.table_names():
            tbl = db.open_table(table_name)
            current_count = len(tbl)

            # AUTOMATIC SKIP: Skip table if already fully ingested
            if current_count >= total_docs:
                logger.info(f"Table '{table_name}' is fully ingested ({current_count} rows). Skipping to save time.")
                continue
            elif current_count > 0:
                logger.warning(f"Table '{table_name}' is partially ingested ({current_count}/{total_docs}).")
                logger.warning(f"Dropping table '{table_name}' to ensure clean ingestion and prevent fragments.")
                db.drop_table(table_name)
                tbl = None

        pending_data = []

        for i in tqdm(range(0, total_docs, batch_size), desc=f"Ingesting {table_name}"):
            batch = subset.iloc[i : i + batch_size]

            # Generate Embeddings (BGE-Large benefits heavily from FP16 speedups)
            texts = batch["content"].fillna("").astype(str).tolist()
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)

            # [OPTIMIZATION 2]: Avoid iterrows(). Converting the batch to a list of dicts
            # eliminates severe CPU bottlenecks, allowing data to stream into the GPU efficiently.
            records = batch.to_dict(orient="records")

            for j, row in enumerate(records):
                pending_data.append(
                    {
                        "doc_id": str(row["doc_id"]),
                        "original_doc_id": str(row["original_doc_id"]),
                        "chunk_id": int(row["chunk_id"]),
                        "content": str(row["content"]),
                        "title": str(row["title"]) if pd.notna(row["title"]) else None,
                        "source_type": str(row["source_type"]),
                        "vector": embeddings[j].tolist(),
                    }
                )

            # DISK WRITE BATCHING: Flush to disk only when reaching write_batch_size to prevent inode exhaustion
            if len(pending_data) >= write_batch_size:
                if tbl is None:
                    tbl = db.create_table(table_name, schema=DocumentSchema, data=pending_data)
                else:
                    tbl.add(pending_data)
                pending_data = []  # Reset buffer after flush

        # Flush any remaining records at the end of the source processing
        if len(pending_data) > 0:
            if tbl is None:
                tbl = db.create_table(table_name, schema=DocumentSchema, data=pending_data)
            else:
                tbl.add(pending_data)
            pending_data = []

    logger.info("LanceDB Ingestion (Sharded) Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise LanceDB Sharding & Ingestion Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/EnterpriseRAG-Bench/data/documents",
        help="Directory containing the chunked parquet file",
    )
    parser.add_argument(
        "--file-name", type=str, default="test_chunked_v2.parquet", help="Filename of the chunked parquet"
    )
    parser.add_argument("--db-path", type=str, default="./data/lancedb", help="Path to the LanceDB storage directory")

    # GPU Batch Size is set to 256 for a 20GB VRAM GPU with FP16 enabled.
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the embedding model")

    # Disk Write Batch Size prevents excessive fragment creation in LanceDB.
    parser.add_argument(
        "--write-batch-size", type=int, default=10000, help="Number of records to accumulate before flushing to disk"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model ID for generating embeddings",
    )

    args = parser.parse_args()
    main(args.data_dir, args.file_name, args.db_path, args.batch_size, args.embedding_model, args.write_batch_size)
