import sys
import time
sys.path.append('/network-volume/RAG-/T-RAG_Project')
from src.retrieval.retriever import EnterpriseRetriever

retriever = EnterpriseRetriever()

q = "What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?"
start = time.time()
res = retriever.retrieve(q, top_k=20)
print(f"Total time 1: {time.time() - start:.3f}s")

start = time.time()
res = retriever.retrieve(q, top_k=20)
print(f"Total time 2: {time.time() - start:.3f}s")
