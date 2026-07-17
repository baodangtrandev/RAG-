import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Constants based on typical project structure
SOURCES_DIR = os.path.join(PROJECT_ROOT, "data", "EnterpriseRAG-Bench", "data", "documents")
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated")
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "questions.json")
AGGREGATE_STATISTICS_PATH = os.path.join(PROJECT_ROOT, "data", "statistics.json")
GENERATION_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
UUID_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "uuid_index.json")
AGENTS_MD_FILE = "AGENTS.md"
