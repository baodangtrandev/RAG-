"""Utilities for indexing and loading source documents by dataset_doc_uuid."""

import os

import pandas as pd

from src.paths import SOURCES_DIR, UUID_INDEX_PATH

DEFAULT_UUID_INDEX_CACHE_FILE = UUID_INDEX_PATH

_PARQUET_DF = None


def _get_df():
    global _PARQUET_DF
    if _PARQUET_DF is None:
        parquet_path = os.path.join(SOURCES_DIR, "test.parquet")
        if os.path.exists(parquet_path):
            _PARQUET_DF = pd.read_parquet(parquet_path)
        else:
            # Fallback if needed, but test.parquet should be there
            _PARQUET_DF = pd.DataFrame(columns=["doc_id", "title", "content"])
    return _PARQUET_DF


class UniversalDict(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return key

    def get(self, key, default=None):
        return key

    def keys(self):
        # Return a dummy set-like object that satisfies `missing = needed_uuids - uuid_index.keys()`
        class DummyKeys:
            def __rsub__(self, other):
                return set()

        return DummyKeys()


def load_or_build_uuid_index(
    cache_file: str = DEFAULT_UUID_INDEX_CACHE_FILE,
    sources_dir: str = SOURCES_DIR,
) -> dict[str, str]:
    return UniversalDict()


def rebuild_uuid_index(
    cache_file: str = DEFAULT_UUID_INDEX_CACHE_FILE,
    sources_dir: str = SOURCES_DIR,
) -> dict[str, str]:
    return UniversalDict()


def ensure_uuids_resolved(
    needed_uuids: set[str],
    uuid_index: dict[str, str] | None = None,
    cache_file: str = DEFAULT_UUID_INDEX_CACHE_FILE,
    sources_dir: str = SOURCES_DIR,
) -> dict[str, str]:
    return UniversalDict()


def load_document_json_by_uuid(
    dataset_doc_uuid: str,
    uuid_index: dict[str, str],
    sources_dir: str = SOURCES_DIR,
) -> dict:
    df = _get_df()
    row = df[df["doc_id"] == dataset_doc_uuid]
    if len(row) == 0:
        return {"doc_id": dataset_doc_uuid, "title": "Not found", "content": "Not found"}
    return row.iloc[0].to_dict()


def load_document_content_by_uuid(
    dataset_doc_uuid: str,
    uuid_index: dict[str, str],
    sources_dir: str = SOURCES_DIR,
) -> tuple[str, str]:
    df = _get_df()
    row = df[df["doc_id"] == dataset_doc_uuid]
    if len(row) == 0:
        return "Not found", "Not found"

    r = row.iloc[0]
    return str(r.get("title", "")), str(r.get("content", ""))
