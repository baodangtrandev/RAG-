from typing import List
from vllm import LLM, SamplingParams

class TRAG_LLMEngine:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size: int = 1):
        print(f"Loading vLLM model {model_name} with TP={tensor_parallel_size}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.6, #60% of GPU memory
            max_model_len=8192
        )
        
    def batch_generate(self, prompts: List[str], max_tokens: int = 512, temperature: float = 0.1) -> List[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        # vLLM naturally batches this entire list via PagedAttention
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract the generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results
