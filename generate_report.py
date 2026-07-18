import json
import glob
import os
import statistics

def parse_config_from_filename(filename):
    name = os.path.basename(filename).replace('.jsonl', '')
    if name.startswith('baseline_'):
        return name.replace('baseline_', '').upper()
    if name.startswith('trag_'):
        config = name.replace('trag_', '')
        if config == 'csep_true': return 'T-RAG (Default: Tau=0.15, Gamma=2.0)'
        if config == 'csep_false': return 'T-RAG (No CSEP)'
        if config.startswith('tau_'): return f'T-RAG (Tau={config.split("_")[1]}, Gamma=2.0)'
        if config.startswith('gamma_'): return f'T-RAG (Tau=0.15, Gamma={config.split("_")[1]})'
        if config.startswith('targeted_high_recall'): return 'T-RAG (Tau=0.05, Gamma=0.0) [High-Recall]'
        if config.startswith('targeted_balanced'): return 'T-RAG (Tau=0.15, Gamma=0.0) [Balanced]'
        if config.startswith('targeted_high_speed'): return 'T-RAG (Tau=0.30, Gamma=0.0) [High-Speed]'
        if config.startswith('targeted_gamma_'): return f'T-RAG (Tau=0.15, Gamma={config.split("_")[2]})'
        return name
    return name

def main():
    print("=" * 110)
    print(f"{'Pipeline / Config':<45} | {'Correct':<8} | {'Complete':<9} | {'Refused':<8} | {'Total Lat':<10} | {'Retr Lat':<10}")
    print("-" * 110)
    
    files = sorted(glob.glob("results/*.jsonl"))
    for file in files:
        filename = os.path.basename(file)
        config_name = parse_config_from_filename(filename)
        
        # Calculate Latency and Refusal Rate
        total_lats = []
        retr_lats = []
        unanswerable = 0
        total = 0
        with open(file, 'r') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    total += 1
                    if row.get('latency_sec') is not None:
                        total_lats.append(row['latency_sec'])
                    if row.get('retrieval_latency_sec') is not None:
                        retr_lats.append(row['retrieval_latency_sec'])
                    
                    ans = row.get('answer', '')
                    if 'do not have enough' in ans.lower() or 'i don' in ans.lower():
                        unanswerable += 1
                except:
                    pass
        
        avg_total_lat = statistics.mean(total_lats) if total_lats else 0
        avg_retr_lat = statistics.mean(retr_lats) if retr_lats else 0
        refused_pct = (unanswerable / total * 100) if total > 0 else 0
        
        # Parse Evaluation Scores
        eval_file = f"results/eval_{filename.replace('.jsonl', '.json')}"
        correct_pct = 0.0
        complete_pct = 0.0
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                try:
                    eval_data = json.load(f)
                    stats = eval_data.get('aggregate_stats', {})
                    correct_pct = float(stats.get('average_correctness_pct', 0))
                    complete_pct = float(stats.get('average_completeness_pct', 0))
                except:
                    pass
        else:
            correct_pct = -1.0 # Means eval not finished
            complete_pct = -1.0
            
        corr_str = f"{correct_pct:.1f}%" if correct_pct >= 0 else "N/A"
        comp_str = f"{complete_pct:.1f}%" if complete_pct >= 0 else "N/A"
        
        print(f"{config_name:<45} | {corr_str:<8} | {comp_str:<9} | {refused_pct:5.1f}%   | {avg_total_lat:5.2f}s     | {avg_retr_lat:5.2f}s")
        
    print("=" * 110)
    print("* Refused = Tỷ lệ LLM sinh ra 'I do not have enough information...'")
    print("* N/A = LLM Judge đang chấm điểm (chưa có file eval json).")

if __name__ == "__main__":
    main()
