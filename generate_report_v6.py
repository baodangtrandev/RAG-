import json
import glob
import os
import statistics
import sys

def parse_config_from_filename(filename):
    name = os.path.basename(filename).replace('.jsonl', '')
    
    # 1. Baselines
    if name == 'baseline_bm25': return 'BM25 Baseline', '1. Baselines'
    if name == 'baseline_vector': return 'VECTOR Baseline', '1. Baselines'
    if name == 'baseline_vector_reranker': return 'VECTOR_RERANKER Baseline', '1. Baselines'
    if name == 'baseline_hybrid': return 'HYBRID Baseline', '1. Baselines'
    if name == 'baseline_hyde': return 'HyDE Baseline', '1. Baselines'
    if name == 'baseline_query_expansion': return 'Query Expansion Baseline', '1. Baselines'
    if name == 'baseline_llm_router': return 'LLM Router Baseline', '1. Baselines'
    
    # 2. T-RAG v1
    if name == 'trag_v1_balanced': return 'T-RAG v1 Balanced (Tau=0.15, Gamma=0.0)', '2. T-RAG v1'
    if name == 'trag_v1_balanced_g1': return 'T-RAG v1 Balanced G1 (Tau=0.15, Gamma=1.0)', '2. T-RAG v1'
    if name == 'trag_v1_high_speed': return 'T-RAG v1 High-Speed (Tau=0.30, Gamma=0.0)', '2. T-RAG v1'
    if name == 'trag_v1_high_speed_g1': return 'T-RAG v1 High-Speed G1 (Tau=0.30, Gamma=1.0)', '2. T-RAG v1'
    if name == 'trag_v1_high_recall': return 'T-RAG v1 High-Recall (Tau=0.05, Gamma=0.0)', '2. T-RAG v1'
    if name == 'trag_v1_high_recall_g1': return 'T-RAG v1 High-Recall G1 (Tau=0.05, Gamma=1.0)', '2. T-RAG v1'
    if name == 'trag_v1_no_reranker': return 'T-RAG v1 No Reranker (Tau=0.15, Gamma=0.0)', '2. T-RAG v1'
    
    # 3. T-RAG v2 Standard
    if name == 'trag_v2_standard': return 'T-RAG v2 Standard (Tau=0.15, G=0.5, Alpha=0.08)', '3. T-RAG v2 Standard'
    
    # 4. T-RAG v2 Grid Search: Tau
    if name == 'trag_v2_grid_tau_0.05': return 'Grid Tau = 0.05 (Min)', '4. T-RAG v2 Grid: Tau Base'
    if name == 'trag_v2_grid_tau_0.10': return 'Grid Tau = 0.10', '4. T-RAG v2 Grid: Tau Base'
    if name == 'trag_v2_grid_tau_0.20': return 'Grid Tau = 0.20', '4. T-RAG v2 Grid: Tau Base'
    if name == 'trag_v2_grid_tau_0.30': return 'Grid Tau = 0.30 (Max)', '4. T-RAG v2 Grid: Tau Base'
    
    # 5. T-RAG v2 Grid Search: Gamma
    if name == 'trag_v2_grid_gamma_0.0': return 'Grid Gamma = 0.0 (Min)', '5. T-RAG v2 Grid: Gamma'
    if name == 'trag_v2_grid_gamma_0.3': return 'Grid Gamma = 0.3', '5. T-RAG v2 Grid: Gamma'
    if name == 'trag_v2_grid_gamma_0.7': return 'Grid Gamma = 0.7', '5. T-RAG v2 Grid: Gamma'
    if name == 'trag_v2_grid_gamma_1.0': return 'Grid Gamma = 1.0 (Max)', '5. T-RAG v2 Grid: Gamma'
    
    # 6. T-RAG v2 Grid Search: Alpha
    if name == 'trag_v2_grid_alpha_0.00': return 'Grid Alpha = 0.00 (Min - static)', '6. T-RAG v2 Grid: Alpha'
    if name == 'trag_v2_grid_alpha_0.04': return 'Grid Alpha = 0.04', '6. T-RAG v2 Grid: Alpha'
    if name == 'trag_v2_grid_alpha_0.12': return 'Grid Alpha = 0.12', '6. T-RAG v2 Grid: Alpha'
    if name == 'trag_v2_grid_alpha_0.15': return 'Grid Alpha = 0.15', '6. T-RAG v2 Grid: Alpha'
    if name == 'trag_v2_grid_alpha_0.25': return 'Grid Alpha = 0.25', '6. T-RAG v2 Grid: Alpha'
    if name == 'trag_v2_grid_alpha_0.50': return 'Grid Alpha = 0.50 (Max - aggressive)', '6. T-RAG v2 Grid: Alpha'
    
    # 7. T-RAG v2 Grid Search: Hybrid Weights
    if name == 'trag_v2_dense_only': return 'Dense Search Only (D=1.0, S=0.0)', '7. T-RAG v2 Grid: Dense/Sparse'
    if name == 'trag_v2_sparse_only': return 'Sparse Search Only (D=0.0, S=1.0)', '7. T-RAG v2 Grid: Dense/Sparse'
    if name == 'trag_v2_hybrid_dense_0.7': return 'Hybrid Dense Heavy (D=0.7, S=0.3)', '7. T-RAG v2 Grid: Dense/Sparse'
    if name == 'trag_v2_hybrid_dense_0.3': return 'Hybrid Sparse Heavy (D=0.3, S=0.7)', '7. T-RAG v2 Grid: Dense/Sparse'
    if name == 'trag_v2_hybrid_dense_0.9': return 'Hybrid Dense Super-Heavy (D=0.9, S=0.1)', '7. T-RAG v2 Grid: Dense/Sparse'
    if name == 'trag_v2_hybrid_dense_0.1': return 'Hybrid Sparse Super-Heavy (D=0.1, S=0.9)', '7. T-RAG v2 Grid: Dense/Sparse'
    
    # 8. T-RAG v2 Ablation Study
    if name == 'trag_v2_ablation_no_smart_hop2': return 'Ablation: No Smart Hop 2 (Hop 2 always run)', '8. T-RAG v2 Ablations'
    if name == 'trag_v2_ablation_no_adaptive_tau': return 'Ablation: No Adaptive Tau (Alpha=0)', '8. T-RAG v2 Ablations'
    if name == 'trag_v2_ablation_no_csep': return 'Ablation: No CSEP (Hop 2 completely skipped)', '8. T-RAG v2 Ablations'
    
    # 9. T-RAG v2 Optimized Combos (v6.1)
    if name == 'opt_best_correctness': return 'OPT: Best Correctness (A=0.25, D=0.1/S=0.9)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_best_completeness': return 'OPT: Best Completeness (A=0.04, D=0.3/S=0.7)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_low_latency': return 'OPT: Low Latency (Tau=0.20, G=0.3, A=0.00)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_balanced': return 'OPT: Balanced (Tau=0.15, G=0.5, A=0.04)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_d02_s08': return 'OPT: Middle Sparse (D=0.2, S=0.8)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_d04_s06': return 'OPT: Middle Balanced (D=0.4, S=0.6)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_gamma_04': return 'OPT: Gamma=0.4 (Unexplored)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_max_performance': return 'OPT: Max Perf (Tau=0.10, A=0.04, D=0.1/S=0.9)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_speed_sparse': return 'OPT: Speed+Sparse (Tau=0.20, A=0.00, D=0.3/S=0.7)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_topk3': return 'OPT: TopK=3 (Standard, K=3)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_topk_retrieve_10': return 'OPT: Retrieve Depth=10 (Standard, R=10)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_topk_retrieve_30': return 'OPT: Retrieve Depth=30 (Standard, R=30)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_topk1': return 'OPT: TopK=1 (Standard, K=1)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_high_recall_sparse_heavy': return 'OPT: Recall+Sparse (Tau=0.05, D=0.3/S=0.7)', '9. T-RAG v2 Optimized (v6.1)'
    if name == 'opt_high_speed_sparse_super_heavy': return 'OPT: Speed+Sparse (Tau=0.30, D=0.1/S=0.9)', '9. T-RAG v2 Optimized (v6.1)'
    
    return name, '10. Custom / Other'

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results_v6"
    files = sorted(glob.glob(f"{folder}/*.jsonl"))
    
    groups = {}
    
    for file in files:
        filename = os.path.basename(file)
        config_name, group_name = parse_config_from_filename(filename)
        
        total_lats = []
        retr_lats = []
        search_spaces = []
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
                    if row.get('search_space_docs') is not None:
                        search_spaces.append(row['search_space_docs'])
                    
                    ans = row.get('answer', '')
                    if 'do not have enough' in ans.lower() or 'i don' in ans.lower():
                        unanswerable += 1
                except:
                    pass
                    
        avg_total_lat = statistics.mean(total_lats) if total_lats else 0
        avg_retr_lat = statistics.mean(retr_lats) if retr_lats else 0
        avg_search_space = statistics.mean(search_spaces) if search_spaces else 0
        refused_pct = (unanswerable / total * 100) if total > 0 else 0
        
        # Parse Evaluation Scores
        eval_file = f"{folder}/eval_{filename.replace('.jsonl', '.json')}"
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
            correct_pct = -1.0
            complete_pct = -1.0
            
        corr_str = f"{correct_pct:.1f}%" if correct_pct >= 0 else "N/A"
        comp_str = f"{complete_pct:.1f}%" if complete_pct >= 0 else "N/A"
        space_str = f"{avg_search_space:,.0f}" if avg_search_space > 0 else "N/A"
        
        if group_name not in groups:
            groups[group_name] = []
            
        groups[group_name].append({
            'config': config_name,
            'correctness': corr_str,
            'completeness': comp_str,
            'refused': refused_pct,
            'total_lat': avg_total_lat,
            'retr_lat': avg_retr_lat,
            'search_space': space_str
        })
        
    for group_name in sorted(groups.keys()):
        print("\n" + "=" * 135)
        print(f" CATEGORY: {group_name.upper()}")
        print("=" * 135)
        print(f"{'Pipeline / Config':<45} | {'Correct':<8} | {'Complete':<9} | {'Refused':<8} | {'Total Lat':<10} | {'Retr Lat':<10} | {'Space Search (Docs)':<20}")
        print("-" * 135)
        
        for row in groups[group_name]:
            print(f"{row['config']:<45} | {row['correctness']:<8} | {row['completeness']:<9} | {row['refused']:5.1f}%   | {row['total_lat']:5.2f}s     | {row['retr_lat']:5.2f}s     | {row['search_space']:<20}")
            
        print("-" * 135)

    print("\n* Refused = Tỷ lệ LLM sinh ra 'I do not have enough information...'")
    print("* N/A = LLM Judge đang chấm điểm (chưa có file eval json).")

if __name__ == "__main__":
    main()
