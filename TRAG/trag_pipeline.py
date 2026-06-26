import json
import math
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
        Stage 1: Ask LLM to extract source_type if any.
        """
        print("Stage 1: Batch Query Parsing...")
        system_prompt = "You are a query analyzer. Extract the target source_type from the query (e.g., 'gmail', 'slack', 'github', 'confluence', 'google_drive', 'linear', 'fireflies'). Return ONLY a valid JSON: {\"source_type\": \"...\"} or null if none."
        
        prompts = []
        for q in questions:
            # Simple Llama-3 format
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
                    parsed_results.append({"source_type": None})
            except Exception:
                parsed_results.append({"source_type": None})
                
        return parsed_results

    def batch_hybrid_retrieval(self, questions: List[str], parsed_queries: List[Dict], top_k: int = 50) -> List[List[Dict]]:
        """
        Stage 2: Execute LanceDB Hybrid Search for each query.
        """
        print("Stage 2: Batch Hybrid Retrieval...")
        all_retrieved = []
        
        for idx, query in enumerate(questions):
            source_type = parsed_queries[idx].get("source_type")
            
            search_builder = self.table.search(query, query_type="hybrid").limit(top_k)
            
            # Apply metadata filtering if detected
            if source_type:
                search_builder = search_builder.where(f"source_type = '{source_type}'", prefilter=True)
                
            try:
                results_df = search_builder.to_pandas()
                docs = results_df.to_dict(orient="records")
            except Exception as e:
                # Fallback without prefiltering if it fails
                try:
                    docs = self.table.search(query, query_type="hybrid").limit(top_k).to_pandas().to_dict(orient="records")
                except:
                    docs = []
                
            all_retrieved.append(docs)
            
        return all_retrieved

    def batch_temporal_reranking(self, questions: List[str], retrieved_docs: List[List[Dict]], top_k: int = 10, lambda_val: float = 0.0) -> List[List[Dict]]:
        """
        Stage 3: CrossEncoder Reranking + Time Decay.
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
        
        for i, score in enumerate(scores):
            q_idx, d_idx = doc_pointers[i]
            doc = retrieved_docs[q_idx][d_idx]
            
            # Apply Temporal Decay (lambda_val=0 by default for EnterpriseRAG-Bench since no timestamps)
            decay_factor = math.exp(-lambda_val * 0) 
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
                context_str += f"Document {idx+1}:\nTitle: {doc.get('title', '')}\nContent: {doc.get('content', '')}\n\n"
                
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context_str}\n\nQuestion: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompts.append(prompt)
            
        outputs = self.llm.batch_generate(prompts, max_tokens=512, temperature=0.1)
        return outputs
