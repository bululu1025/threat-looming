"""
Project: Hanmengfan Looming Fear - fPCA Module (Updated for Excel Demo Data)
Target Journal: PLOS Biology

Expert Note:
This script performs Functional Principal Component Analysis (FPCA) with DYNAMIC component selection.
It automatically selects the number of Principal Components (PCs) needed to explain at least 85% 
of the total variance.
"""
import pandas as pd
import numpy as np
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from scipy.interpolate import interp1d
import os

def create_functional_data(group, n_points=200):
    """
    long format data to functional data vector.
    ensure matrix dimension strictly consistent (N x 200).
    """
    if len(group) < 2:
        return None
    
    # ensure sorted by time
    group = group.sort_values('normalized_time')
    
    # create linear interpolation function
    try:
        f = interp1d(group['normalized_time'], group['diameter'], kind='linear')
    except KeyError:
        # try to compatible with old column names, prevent error
        f = interp1d(group['normalized_time'], group['diameter_corrected'], kind='linear')
    
    # generate strictly uniform grid points
    uniform_time = np.linspace(0, 1, n_points)
    
    # resample
    uniform_diameter = f(uniform_time)
    
    return uniform_diameter

def perform_fpca(df):
    """perform FPCA analysis and automatically select PC number based on 85% variance threshold"""
    n_points = 200
    grid_points = np.linspace(0, 1, n_points)
    target_variance_ratio = 0.85  # set threshold to 85%
    
    fd_list = []
    metadata = []
    
    print("building functional data matrix (Long-to-Wide Conversion)...")
    
    # ensure id and trialIndex are integers or strings, prevent float error
    df['id'] = df['id'].astype(str)
    df['trialIndex'] = df['trialIndex'].astype(int)

    for name, group in df.groupby(['id', 'trialIndex']):
        try:
            fd = create_functional_data(group, n_points)
            if fd is not None:
                fd_list.append(fd)
                # get metadata, use iloc[0] to get the first row of the group
                metadata.append({
                    'id': name[0],
                    'trialIndex': name[1],
                    'treatment': group['treatment'].iloc[0],
                    'gender': group['gender'].iloc[0],
                    'Isthreaten': group['Isthreaten'].iloc[0],
                    'PSV': group['PSV'].iloc[0],
                    'RT': group['RT'].iloc[0] 
                })
        except Exception as e:
            # print specific error for debugging
            # print(f"Skipping trial {name}: {e}")
            continue
    
    if not fd_list:
        raise ValueError("cannot create valid functional data, please check input data format or column names.")
    
    fd_matrix = np.stack(fd_list)
    print(f"data matrix built: {fd_matrix.shape} (Trials x Timepoints)")
    
    fd = FDataGrid(fd_matrix, grid_points)
    
    # --- dynamically select PC ---
    # first compute the first 10 components (if sample size is less than 10, take sample size)
    max_components_to_compute = min(10, fd_matrix.shape[0] - 1) 
    if max_components_to_compute < 1: max_components_to_compute = 1
    
    print(f"calculating the first {max_components_to_compute} components to determine the optimal number...")
    
    fpca_full = FPCA(n_components=max_components_to_compute)
    fpca_full.fit(fd)
    
    # get explained variance
    explained_variance = fpca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # find the cutoff point that reaches 85%
    # np.argmax 返回第一个 True 的索引
    n_components_optimal = np.argmax(cumulative_variance >= target_variance_ratio) + 1
    
    print("\n" + "="*50)
    print(f"FPCA dynamically selected results (threshold: {target_variance_ratio*100}%)")
    print("="*50)
    for i in range(len(explained_variance)):
        print(f"PC{i+1}: {explained_variance[i]*100:.2f}% (cumulative: {cumulative_variance[i]*100:.2f}%)")
            
    print("-" * 50)
    print(f"conclusion: need to retain the first {n_components_optimal} components to explain >{target_variance_ratio*100}% variance.")
    print(f"actual explained variance: {cumulative_variance[n_components_optimal-1]*100:.2f}%")
    print("="*50 + "\n")
    
    # re-convert only the needed components 
    scores_all = fpca_full.transform(fd)
    scores_optimal = scores_all[:, :n_components_optimal]
    
    # prepare output data
    trial_data = []
    for i in range(len(fd_list)):
        trial_info = metadata[i].copy()
        # save scores as list
        trial_info['PC_scores'] = scores_optimal[i].tolist()
        trial_data.append(trial_info)
    
    return {
        'fpca_scores': scores_optimal,
        'trial_data': trial_data,
        'grid_points': grid_points,
        'fpca_object': fpca_full,
        'n_components': n_components_optimal
    }

def main():
    # update file path and read method (relative to this script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'demo_data.xlsx')
    sheet_name = 'pupil'
    
    print(f"reading data: {input_path} (Sheet: {sheet_name})")
    
    # use read_excel instead of read_csv
    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print("error: file not found, please check the path.")
        return
    except Exception as e:
        print(f"reading Excel failed: {e}")
        return
    
    # perform analysis
    try:
        results = perform_fpca(df)
    except Exception as e:
        print(f"error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # save results
    print("saving results...")
    
    data_dict = {
        'id': [t['id'] for t in results['trial_data']],
        'trialIndex': [t['trialIndex'] for t in results['trial_data']],
        'treatment': [t['treatment'] for t in results['trial_data']],
        'gender': [t['gender'] for t in results['trial_data']],
        'Isthreaten': [t['Isthreaten'] for t in results['trial_data']],
        'PSV': [t['PSV'] for t in results['trial_data']],
        'RT': [t['RT'] for t in results['trial_data']]
    }
    
    # dynamically add PC Scores column
    scores = results['fpca_scores']
    n_comps = results['n_components']
    
    for i in range(n_comps):
        data_dict[f'PC{i+1}_Score'] = scores[:, i]
        
    df_out = pd.DataFrame(data_dict)
    
    # output path setting (save in the same directory as the input file)
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, 'FPCA_Results_Demo.csv')
    
    df_out.to_csv(output_path, index=False)
    print(f"PC scores have been saved to: {output_path}")
    
    # save eigenfunctions
    eigenfunctions = results['fpca_object'].components_.data_matrix
    eigen_df = pd.DataFrame({'normalized_time': results['grid_points']})
    
    # note: if the number of extracted PCs is less than the number of computed, here to prevent overflow
    for i in range(n_comps):
        if i < len(eigenfunctions):
            eigen_df[f'PC{i+1}_Weight'] = eigenfunctions[i].flatten()
        
    eigen_path = os.path.join(output_dir, 'FPCA_Eigenfunctions_Demo.csv')
    eigen_df.to_csv(eigen_path, index=False)
    print(f"eigenfunctions have been saved to: {eigen_path}")

if __name__ == "__main__":
    main()