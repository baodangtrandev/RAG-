import lancedb
from lancedb.pydantic import LanceModel, Vector

class DocumentSchema(LanceModel):
    doc_id: str
    original_doc_id: str
    chunk_id: int
    content: str
    title: str | None = None
    source_type: str
    vector: Vector(1024)  # bge-large-en-v1.5 dimension
