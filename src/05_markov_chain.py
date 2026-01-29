'''
Looming Fear - Markov Chain Analysis
Author: Hanmengfan
This script is used to analyze the transition matrix and steady state distribution of the pupil diameter data.
Target Journal: PLOS Biology
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as multi
import os
import sys

# ==========================================
# Setting Area
# ==========================================
# Input file path (relative to this script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'demo_data.xlsx')
sheet_name = 'Markov Chain' 

# ==========================================
# Core Calculation Functions
# ==========================================

def calculate_transition_matrix(data, cluster_col='Cluster', id_col='id', predefined_clusters=None):
    """
    Calculate transition matrix and frequency matrix
    :param predefined_clusters: Force specified cluster list to prevent matrix dimension inconsistency when Bootstrap sampling loses states
    """
    if predefined_clusters is not None:
        clusters = predefined_clusters
    else:
        clusters = sorted(data[cluster_col].unique())
        
    n_clusters = len(clusters)
    transition_counts = np.zeros((n_clusters, n_clusters))

    # create a mapping to speed up lookup
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}

    for subject_id, subject_data in data.groupby(id_col):
        subject_data = subject_data.sort_values('trialIndex')
        cluster_sequence = subject_data[cluster_col].values

        for i in range(len(cluster_sequence) - 1):
            current_cluster = cluster_sequence[i]
            next_cluster = cluster_sequence[i + 1]
            
            # Only calculate when both states are in the predefined list (to prevent abnormal values)
            if current_cluster in cluster_to_idx and next_cluster in cluster_to_idx:
                current_idx = cluster_to_idx[current_cluster]
                next_idx = cluster_to_idx[next_cluster]
                transition_counts[current_idx, next_idx] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = transition_counts / row_sums
    transition_matrix = np.nan_to_num(transition_matrix, nan=0)

    return transition_matrix, transition_counts, clusters

def calculate_steady_state(transition_matrix):
    """calculate steady state distribution"""
    n = transition_matrix.shape[0]
    # If the matrix is all 0 (extreme case), return uniform distribution
    if np.all(transition_matrix == 0):
        return np.ones(n) / n

    distribution = np.ones(n) / n
    max_iter = 1000
    epsilon = 1e-8
    
    for i in range(max_iter):
        new_distribution = np.dot(distribution, transition_matrix)
        if np.all(np.abs(new_distribution - distribution) < epsilon):
            break
        distribution = new_distribution
    
    # Normalize to prevent numerical errors
    if np.sum(distribution) == 0:
        return np.ones(n) / n
    return distribution / np.sum(distribution)

def bootstrap_transition_matrix(data, cluster_col='Cluster', n_bootstrap=1000):
    """Bootstrap estimate transition matrix (Subject-level Resampling)"""
    unique_ids = data['id'].unique()
    bootstrap_matrices = []
    
    # === Pre-obtain all global clusters to ensure matrix dimension consistency ===
    global_clusters = sorted(data[cluster_col].unique())
    # ==========================================================

    grouped_data = dict(tuple(data.groupby('id')))

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap Resampling", leave=False):
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        
        df_list = []
        for i, uid in enumerate(sampled_ids):
            sub_df = grouped_data[uid].copy() 
            sub_df['bootstrap_temp_id'] = i
            df_list.append(sub_df)
        
        bootstrap_df = pd.concat(df_list)
        
        # pass in predefined_clusters
        transition_matrix, _, _ = calculate_transition_matrix(
            bootstrap_df, cluster_col, id_col='bootstrap_temp_id', 
            predefined_clusters=global_clusters
        )
        bootstrap_matrices.append(transition_matrix)

    # Convert to numpy array to ensure correct dimensions
    try:
        bootstrap_matrices = np.array(bootstrap_matrices)
        mean_matrix = np.mean(bootstrap_matrices, axis=0)
    except ValueError as e:
        print("Error during matrix averaging. Check dimensions.")
        raise e
        
    return mean_matrix, bootstrap_matrices

def bootstrap_steady_state(bootstrap_matrices, clusters):
    """Bootstrap estimate steady state distribution"""
    steady_states = []
    for matrix in bootstrap_matrices:
        ss = calculate_steady_state(matrix)
        steady_states.append(ss)
    
    steady_states = np.array(steady_states)
    mean_ss = np.mean(steady_states, axis=0)
    lower_bound = np.percentile(steady_states, 2.5, axis=0)
    upper_bound = np.percentile(steady_states, 97.5, axis=0)
    
    return pd.DataFrame({
        'Cluster': clusters,
        'Probability': mean_ss,
        'Lower_CI': lower_bound,
        'Upper_CI': upper_bound
    }), steady_states

# ==========================================
# Between-group comparison functions
# ==========================================

def compare_steady_states(group_results, alpha=0.05):
    """Compare steady state distribution differences"""
    levels = list(group_results.keys())
    if not levels: return
    
    clusters = group_results[levels[0]]['steady_state_df']['Cluster'].tolist()
    
    print(f"\n{'='*80}")
    print(f"PAIRWISE COMPARISONS: STEADY STATE DISTRIBUTION")
    print(f"{'='*80}")
    
    comparison_pairs = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            comparison_pairs.append((levels[i], levels[j]))
            
    for level1, level2 in comparison_pairs:
        print(f"\nComparing: {level2} vs {level1}")
        print(f"{'-'*50}")
        
        ss_samples1 = group_results[level1]['steady_state_samples']
        ss_samples2 = group_results[level2]['steady_state_samples']
        
        results_list = []
        
        for i, cluster in enumerate(clusters):
            # extract all bootstrap samples of this cluster
            vals1 = ss_samples1[:, i]
            vals2 = ss_samples2[:, i]
            
            diffs = vals2 - vals1
            mean_diff = np.mean(diffs)
            
            # calculate p-value
            if mean_diff > 0:
                p_val = 2 * (1 - (np.sum(diffs > 0) / len(diffs)))
            else:
                p_val = 2 * (np.sum(diffs > 0) / len(diffs))
            p_val = max(min(p_val, 1.0), 0.001)
            
            results_list.append({
                'Cluster': cluster,
                'mean_diff': mean_diff,
                'p': p_val
            })
            
        # FDR correction
        p_vals = [r['p'] for r in results_list]
        _, p_adj, _, _ = multi.multipletests(p_vals, alpha=alpha, method='fdr_bh')
        
        has_sig = False
        for i, res in enumerate(results_list):
            res['p_adj'] = p_adj[i]
            if res['p'] < alpha: # Show uncorrected significant results as a trend
                sig_mark = "**" if res['p_adj'] < alpha else "*"
                direction = "higher" if res['mean_diff'] > 0 else "lower"
                print(f"State {res['Cluster']}: {level2} is {abs(res['mean_diff']):.3f} {direction} (p={res['p']:.3f}, p_adj={res['p_adj']:.3f}) {sig_mark}")
                has_sig = True
        
        if not has_sig:
            print("No significant differences found.")

def compare_transition_matrices(group_results, alpha=0.05, min_prob_threshold=0.03):
    """
    Compare transition matrix differences
    """
    levels = list(group_results.keys())
    if not levels: return []

    clusters = group_results[levels[0]]['steady_state_df']['Cluster'].tolist()
    n_clusters = len(clusters)
    all_comparisons = []
    
    comparison_pairs = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            comparison_pairs.append((levels[i], levels[j]))
    
    print(f"\n{'='*80}")
    print(f"PAIRWISE COMPARISONS: TRANSITION MATRICES (Filtered)")
    print(f"Filter: Testing only if prob > {min_prob_threshold} in at least one group.")
    print(f"NOTE: ** = FDR Corrected Significant; * = Uncorrected Significant (p<0.05)")
    print(f"{'='*80}")

    for level1, level2 in comparison_pairs:
        print(f"\nComparing: {level2} vs {level1}")
        print(f"{'-'*50}")
        
        comparison_data = []
        matrices1 = group_results[level1]['bootstrap_matrices']
        matrices2 = group_results[level2]['bootstrap_matrices']
        
        skipped_count = 0
        
        for from_idx in range(n_clusters):
            for to_idx in range(n_clusters):
                from_state = clusters[from_idx]
                to_state = clusters[to_idx]
                
                probs1 = matrices1[:, from_idx, to_idx]
                probs2 = matrices2[:, from_idx, to_idx]
                
                # === New filtering logic ===
                mean_p1 = np.mean(probs1)
                mean_p2 = np.mean(probs2)
                
                if mean_p1 <= min_prob_threshold and mean_p2 <= min_prob_threshold:
                    skipped_count += 1
                    continue
                # ===================
                
                diff_bootstrap = probs2 - probs1
                mean_diff = np.mean(diff_bootstrap)
                
                if mean_diff > 0:
                    p_val = 2 * (1 - (np.sum(diff_bootstrap > 0) / len(diff_bootstrap)))
                else:
                    p_val = 2 * (np.sum(diff_bootstrap > 0) / len(diff_bootstrap))
                p_val = max(min(p_val, 1.0), 0.001)
                
                comparison_data.append({
                    'from': from_state, 'to': to_state, 'p': p_val,
                    'mean_diff': mean_diff
                })

        # Print filtering information
        n_tests = len(comparison_data)
        print(f"Performed {n_tests} tests (Skipped {skipped_count} rare transitions).")

        # FDR correction
        if n_tests > 0:
            p_vals = [item['p'] for item in comparison_data]
            _, p_adjusted, _, _ = multi.multipletests(p_vals, alpha=alpha, method='fdr_bh')
            
            significant_list = []
            for i, item in enumerate(comparison_data):
                item['p_adjusted'] = p_adjusted[i]
                item['is_fdr_sig'] = item['p_adjusted'] < alpha
                item['is_uncorr_sig'] = item['p'] < alpha
                
                if item['is_uncorr_sig']:
                    significant_list.append(item)
            
            significant_list.sort(key=lambda x: abs(x['mean_diff']), reverse=True)
            
            all_comparisons.append({
                'level1': level1, 'level2': level2,
                'significant_differences': significant_list
            })
            
            if significant_list:
                for diff in significant_list:
                    direction = "higher" if diff['mean_diff'] > 0 else "lower"
                    sig_symbol = "**" if diff['is_fdr_sig'] else "*"
                    print(f"{diff['from']} → {diff['to']}: {level2} is {abs(diff['mean_diff']):.3f} {direction} "
                          f"(p={diff['p']:.3f}, p_adj={diff['p_adjusted']:.3f}) {sig_symbol}")
            else:
                print("No significant differences found.")
        else:
            print("No transitions met the probability threshold.")
            all_comparisons.append({'level1': level1, 'level2': level2, 'significant_differences': []})

    return all_comparisons

# ==========================================
# Main analysis logic
# ==========================================

def analyze_treatment_effect(data, n_bootstrap=1000):
    group_results = {}
    
    # Get global unique Cluster list to ensure all groups and all Bootstrap samples use the same dimension
    global_clusters = sorted(data['Cluster'].unique())
    print(f"Global Clusters identified: {global_clusters}")
    
    factor_values = ['LT', 'PLC', 'AVP'] 
    available_values = [v for v in factor_values if v in data['treatment'].unique()]
    observed_counts = pd.DataFrame(index=available_values, columns=global_clusters)
    # Initialize as float type to avoid FutureWarning
    observed_counts = observed_counts.astype(float)

    for treatment in available_values:
        group_data = data[data['treatment'] == treatment]
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR TREATMENT GROUP: {treatment}")
        print(f"{'='*80}")
        
        # 1. Calculate transition matrix (pass in global Cluster)
        _, transition_counts_matrix, _ = calculate_transition_matrix(
            group_data, id_col='id', predefined_clusters=global_clusters
        )
        
        # 2. Bootstrap (force using global Cluster)
        unique_ids = group_data['id'].unique()
        bootstrap_matrices = []
        grouped_data = dict(tuple(group_data.groupby('id')))
        
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap Resampling", leave=False):
            sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
            df_list = []
            for i, uid in enumerate(sampled_ids):
                sub_df = grouped_data[uid].copy() 
                sub_df['bootstrap_temp_id'] = i
                df_list.append(sub_df)
            bootstrap_df = pd.concat(df_list)
            
            # Here force pass in global_clusters
            tm, _, _ = calculate_transition_matrix(
                bootstrap_df, id_col='bootstrap_temp_id', predefined_clusters=global_clusters
            )
            bootstrap_matrices.append(tm)
            
        bootstrap_matrices = np.array(bootstrap_matrices)
        mean_matrix = np.mean(bootstrap_matrices, axis=0)
        
        # 3. Steady state distribution
        steady_state_df, steady_state_samples = bootstrap_steady_state(bootstrap_matrices, global_clusters)
        
        print("\nSteady State Distribution (Mean ± 95% CI):")
        print(f"{'State':<10} {'Probability':<12} {'95% CI':<20}")
        print("-" * 45)
        for _, row in steady_state_df.iterrows():
            print(f"{row['Cluster']:<10} {row['Probability']:.4f}       [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}]")
        
        group_results[treatment] = {
            'mean_matrix': mean_matrix,
            'steady_state_df': steady_state_df,
            'steady_state_samples': steady_state_samples, 
            'bootstrap_matrices': bootstrap_matrices,
            'transition_counts': transition_counts_matrix
        }
        
        for cluster in global_clusters:
            observed_counts.loc[treatment, cluster] = group_data[group_data['Cluster'] == cluster].shape[0]

    return group_results, observed_counts

def main():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    if INPUT_FILE.endswith('.xlsx'):
        try:
            df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
            print(f"Successfully loaded sheet: '{sheet_name}'")
        except ValueError:
            print(f"Error: Sheet '{sheet_name}' not found in the Excel file.")
            xl = pd.ExcelFile(INPUT_FILE)
            print(f"Available sheets are: {xl.sheet_names}")
            return
    else:
        df = pd.read_csv(INPUT_FILE)

    print(f"Columns found: {df.columns.tolist()}")

    treatment_map = {1: 'LT', 2: 'PLC', 3: 'AVP'}
    if 'treatment' in df.columns:
        if df['treatment'].dtype != object:
            df['treatment'] = df['treatment'].map(treatment_map)
        df = df[df['treatment'].isin(['LT', 'PLC', 'AVP'])]
    else:
        print("Error: 'treatment' column not found.")
        return
    
    if 'Cluster' not in df.columns:
        print("Error: 'Cluster' column not found.")
        return
    
    # Ensure Cluster column has no NaN
    df = df.dropna(subset=['Cluster'])
    
    n_bootstrap = 1000 
    results, observed_counts = analyze_treatment_effect(df, n_bootstrap)
    clusters = sorted(df['Cluster'].unique())

    # Global chi-square test
    print(f"\n{'='*80}")
    print("GLOBAL CHI-SQUARE TESTS")
    print(f"{'='*80}")
    
    # Fix FutureWarning: Use fillna(0) and ensure correct type
    observed_counts = observed_counts.fillna(0)
    chi2_ss, p_ss, dof_ss, _ = chi2_contingency(observed_counts)
    print(f"Steady State Distribution: χ²={chi2_ss:.2f}, p={p_ss:.4f}")
    
    n_groups = len(results)
    n_transitions = len(clusters) ** 2
    contingency_table = np.zeros((n_groups, n_transitions))
    for i, treatment in enumerate(results.keys()):
        contingency_table[i, :] = results[treatment]['transition_counts'].flatten()
    
    # === Fix ValueError: Remove columns that are all 0 (i.e., never occurred transitions)===
    # If a column is all 0 in all groups, it will cause the expected frequency to be 0, causing an error
    valid_columns = contingency_table.sum(axis=0) > 0
    filtered_contingency_table = contingency_table[:, valid_columns]
    
    if filtered_contingency_table.shape[1] > 0:
        chi2_tm, p_tm, dof_tm, _ = chi2_contingency(filtered_contingency_table)
        print(f"Transition Matrices:     χ²={chi2_tm:.2f}, p={p_tm:.4f}")
    else:
        print("Transition Matrices:     Cannot compute (No valid transitions observed).")

    # Compare steady state distribution
    compare_steady_states(results)

    # Compare transition matrix (only print significant results, no plotting)
    compare_transition_matrices(results, min_prob_threshold=0.01)
    
    print("\nAnalysis Completed.")

if __name__ == "__main__":
    main()