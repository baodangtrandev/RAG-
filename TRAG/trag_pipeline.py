import json
import math
import time
import lancedb
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class TRAGPipeline:
    def __init__(self, db_path: str, llm_engine, reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("enterprise_docs")
        self.llm = llm_engine
        
        print(f"Loading CrossEncoder: {reranker_model}...")
        # CrossEncoder uses GPU automatically if torch detects it
        self.reranker = CrossEncoder(reranker_model)
        
    def batch_parse_queries(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Stage 1: Ask LLM to extract source_type and requires_latest flag.
        """
        print("Stage 1: Batch Query Parsing & Expansion...")
        system_prompt = "You are a query analyzer for an Enterprise RAG system. Your task is to analyze the user's question and output a JSON with 3 fields: 1. 'source_type' (e.g., 'gmail', 'slack', 'github', 'confluence', 'google_drive', 'linear', 'fireflies', or null). 2. 'requires_latest' (true if the query asks for 'new', 'latest', 'current', 'recent' info, else false). 3. 'search_query' (an optimized search string containing keywords, expanded acronyms, and synonyms to maximize retrieval recall). Return ONLY valid JSON: {\"source_type\": \"...\", \"requires_latest\": false, \"search_query\": \"...\"}"
        
        prompts = []
        for q in questions:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompts.append(prompt)
            
        outputs = self.llm.batch_generate(prompts, max_tokens=64, temperature=0.0)
        
        parsed_results = []
        for out in outputs:
            try:
                # Find the first json-like `{...}`
                start = out.find('{')
                end = out.rfind('}')
                if start != -1 and end != -1:
                    data = json.loads(out[start:end+1])
                    parsed_results.append(data)
                else:
                    parsed_results.append({"source_type": None, "requires_latest": False, "search_query": q})
            except Exception:
                parsed_results.append({"source_type": None, "requires_latest": False, "search_query": q})
                
        return parsed_results

    def batch_hybrid_retrieval(self, questions: List[str], parsed_queries: List[Dict], top_k: int = 50) -> List[List[Dict]]:
        """
        Stage 2: Execute LanceDB Hybrid Search for each query.
        """
        print("Stage 2: Batch Hybrid Retrieval...")
        all_retrieved = []
        
        for idx, original_query in enumerate(questions):
            parsed = parsed_queries[idx]
            source_type = parsed.get("source_type")
            # Use the expanded query for retrieval if available, else fallback to original
            search_query = parsed.get("search_query", original_query)
            
            search_builder = self.table.search(search_query, query_type="hybrid").limit(top_k)
            
            # Apply metadata filtering if detected
            if source_type:
                search_builder = search_builder.where(f"source_type = '{source_type}'", prefilter=True)
                
            try:
                results_df = search_builder.to_pandas()
                docs = results_df.to_dict(orient="records")
            except Exception as e:
                # Fallback without prefiltering if it fails
                try:
                    docs = self.table.search(search_query, query_type="hybrid").limit(top_k).to_pandas().to_dict(orient="records")
                except:
                    docs = []
                
            all_retrieved.append(docs)
            
        return all_retrieved

    def batch_temporal_reranking(self, questions: List[str], retrieved_docs: List[List[Dict]], parsed_queries: List[Dict], top_k: int = 10) -> List[List[Dict]]:
        """
        Stage 3: CrossEncoder Reranking + Conditional Time Decay.
        """
        print("Stage 3: Batch Temporal Reranking...")
        pairs = []
        doc_pointers = [] # To map flat scores back to structured list
        
        for q_idx, q in enumerate(questions):
            docs = retrieved_docs[q_idx]
            for d_idx, doc in enumerate(docs):
                pairs.append((q, doc['content']))
                doc_pointers.append((q_idx, d_idx))
                
        if not pairs:
            return [[] for _ in questions]
            
        # Batch predict all at once on GPU (massive speedup)
        scores = self.reranker.predict(pairs, batch_size=256)
        
        final_results = [[] for _ in questions]
        
        current_time = time.time()
        
        for i, score in enumerate(scores):
            q_idx, d_idx = doc_pointers[i]
            doc = retrieved_docs[q_idx][d_idx]
            
            # Apply Temporal Decay only if user wants latest information
            requires_latest = parsed_queries[q_idx].get("requires_latest", False)
            decay_factor = 1.0
            
            if requires_latest and doc.get("timestamp", 0.0) > 0:
                # delta_days = days since document was created
                delta_days = max(0, (current_time - doc["timestamp"]) / 86400)
                # lambda_val = 0.01 (loses 1% score every day)
                decay_factor = math.exp(-0.01 * delta_days)
            
            final_score = score * decay_factor
            
            doc['final_score'] = float(final_score)
            final_results[q_idx].append(doc)
            
        # Sort and take top_k
        top_k_results = []
        for docs in final_results:
            docs.sort(key=lambda x: x['final_score'], reverse=True)
            top_k_results.append(docs[:top_k])
            
        return top_k_results

    def batch_generate_answers(self, questions: List[str], top_k_docs: List[List[Dict]]) -> List[str]:
        """
        Stage 4: Batch Answer Generation
        """
        print("Stage 4: Batch Answer Generation...")
        prompts = []
        system_prompt = "You are a helpful assistant. Use ONLY the provided context to answer the question. Do not hallucinate."
        
        for i, q in enumerate(questions):
            context_str = ""
            for idx, doc in enumerate(top_k_docs[i]):
                doc_str = f"Document {idx+1}:\nTitle: {doc.get('title', '')}\nContent: {doc.get('content', '')}\n\n"
                # Truncate context_str roughly to prevent vLLM max_model_len overflow (assume ~3.5 chars/token, leave ~1000 tokens for prompt/generation)
                if len(context_str) + len(doc_str) > 25000:
                    break
                context_str += doc_str
                
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context_str}\n\nQuestion: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompts.append(prompt)
            
        outputs = self.llm.batch_generate(prompts, max_tokens=512, temperature=0.1)
        return outputs
