''''
HanMengfan coding for PLOS Biology 20260128
Data processing and analysis for pupil diameter data.
This script performs the following steps:
1. Load and preprocess the raw JSONL data.
2. Normalize the time axis of each trial to a standard [0, 1] interval.
3. Calculate quality metrics.
4. Plot the average pupil response curve.
'''
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.stats import pearsonr
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

#######################################################
# Data Loading and Preprocessing
#######################################################

def load_and_preprocess_data(file_path):
    """
    Load and perform initial preprocessing on the raw JSONL data.
    
    Args:
        file_path (str): Path to the input .jsonl file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe sorted by ID and timestamp.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_json(file_path, lines=True)
    
    # Ensure numeric types for critical columns
    df['diameter'] = pd.to_numeric(df['diameter'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['diameter', 'timestamp'])
    
    # Sort data to ensure chronological order for each trial
    df = df.sort_values(['id', 'trialIndex', 'timestamp'])
    return df

def quality_control_autocorrelation(group, threshold=0.5):
    """
    Quality control based on lag-1 autocorrelation.
    High autocorrelation indicates a continuous physiological signal, 
    whereas low autocorrelation suggests noise or tracking loss.
    
    Args:
        group (pd.DataFrame): Data for a single trial.
        threshold (float): Minimum acceptable autocorrelation coefficient.
        
    Returns:
        bool: True if the trial passes QC, False otherwise.
    """
    diameter_series = group['diameter'].copy()
    diameter_series = diameter_series.dropna()
    diameter_series = diameter_series.reset_index(drop=True)
    
    if len(diameter_series) < 2:
        return False
    
    # Calculate Pearson correlation between signal[t] and signal[t+1]
    corr = pearsonr(diameter_series[:-1], diameter_series[1:])[0]
    return corr >= threshold

def normalize_trial_time(df, min_trial_duration=2000):
    """
    Normalize the time axis of each trial to a standard [0, 1] interval.
    
    Methodology:
    1. Segmentation: Each trial is divided into 'Stimulus' and 'Imagination' phases.
    2. Time Warping: 
       - Stimulus phase is mapped to normalized time [0, 0.5].
       - Imagination phase is mapped to normalized time [0.5, 1.0].
    3. Smoothing: A B-Spline basis is used to fit the raw pupil diameter data 
       to handle irregular sampling rates and smooth high-frequency noise.
    
    Args:
        df (pd.DataFrame): Preprocessed raw data.
        min_trial_duration (int): Minimum duration (ms) for a trial to be included.
        
    Returns:
        pd.DataFrame: Time-normalized data containing all valid trials.
    """
    grouped = df.groupby(['id', 'trialIndex'])
    normalized_trials = []
    total_trials = len(grouped)
    skipped_trials = 0
    
    # Define resolution for the normalized time axis
    n_points_stimulus = 100   
    n_points_imagination = 100
    
    print("Processing data and normalizing time axes...")
    
    for (subject, trial), group in grouped:
        try:
            group = group.copy()
            group['timestamp'] = pd.to_numeric(group['timestamp'], errors='coerce')
            group['diameter'] = pd.to_numeric(group['diameter'], errors='coerce')
            
            # Check for necessary event markers (keypress indicating end of imagination)
            if 'keypressed' not in group['state'].values:
                skipped_trials += 1
                continue
            
            # Define temporal landmarks
            t_baseline_start = group['timestamp'].min()
            t_stimulus_start = t_baseline_start + 300  # Offset from baseline
            t_stimulus_end = t_baseline_start + 1300   # Fixed duration for stimulus
            t_imagination_end = group[group['state'] == 'keypressed']['timestamp'].iloc[0]
            
            # Filter based on data quantity
            if len(group) < 10:
                skipped_trials += 1
                continue
            
            # Filter based on total duration
            trial_duration = t_imagination_end - t_stimulus_start
            if trial_duration < min_trial_duration:
                skipped_trials += 1
                continue
            
            # Segment data into phases
            stimulus_data = group[(group['timestamp'] >= t_stimulus_start) & 
                                (group['timestamp'] < t_stimulus_end)].copy()
            imagination_data = group[(group['timestamp'] >= t_stimulus_end) & 
                                  (group['timestamp'] <= t_imagination_end)].copy()
            
            if (len(stimulus_data) < 3 or len(imagination_data) < 3):
                skipped_trials += 1
                continue
            
            # Apply autocorrelation quality control
            if not quality_control_autocorrelation(group):
                skipped_trials += 1
                continue
            
            # Create the target normalized time vector
            normalized_times = np.concatenate([
                np.linspace(0, 0.5, n_points_stimulus),
                np.linspace(0.5, 1.0, n_points_imagination)
            ])
            
            trial_data = []
            
            # Combine phases for continuous spline fitting
            phase_data = pd.concat([stimulus_data, imagination_data])
            
            if len(phase_data) > 0:
                valid_data = phase_data.dropna(subset=['diameter', 'timestamp'])
                if len(valid_data) < 2:
                    raise ValueError(f"Insufficient valid data points for Trial {trial}")
                
                # Calculate time relative to stimulus onset
                relative_time = valid_data['timestamp'] - t_stimulus_start
                
                # --- B-Spline Fitting ---
                # Construct a B-spline basis (df=10, degree=3) to model the pupil response.
                # This provides a smooth approximation of the underlying signal.
                spline_basis = dmatrix("bs(x, df=10, degree=3)", {"x": relative_time})
                
                # Solve for spline coefficients using Least Squares
                spline_coef, _, _, _ = np.linalg.lstsq(spline_basis, valid_data['diameter'], rcond=None)
                spline_fit = spline_basis.dot(spline_coef)
                
                # Create an interpolation function from the fitted spline
                f = interpolate.interp1d(relative_time,
                                         spline_fit, 
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                
                # --- Resampling to Normalized Time ---
                for t in normalized_times:
                    if t <= 0.5:
                        # Map normalized [0, 0.5] back to original Stimulus duration
                        original_time = t_stimulus_start + t * (t_stimulus_end - t_stimulus_start) / 0.5
                        phase = 'stimulus'
                    else:  
                        # Map normalized [0.5, 1.0] back to original Imagination duration
                        # Factor 2 is derived from 1.0 / (1.0 - 0.5)
                        original_time = t_stimulus_end + (t - 0.5) * 2 * (t_imagination_end - t_stimulus_end)
                        phase = 'imagination'
                    
                    interpolated_value = f(original_time - t_stimulus_start)
                    
                    if np.isnan(interpolated_value) or np.isinf(interpolated_value):
                        continue
                    
                    trial_data.append({
                        'id': subject,
                        'trialIndex': trial,
                        'normalized_time': float(t),
                        'original_time': float(original_time),
                        'diameter': float(interpolated_value),
                        'Phase': phase,
                        # Preserve metadata
                        'treatment': group['treatment'].iloc[0],
                        'gender': group['gender'].iloc[0],
                        'category': group['category'].iloc[0],
                        'Isthreaten': group['Isthreaten'].iloc[0],
                        'TTC': group['TTC'].iloc[0],
                        'RT': group['RT'].iloc[0],
                        'PSV': group['PSV'].iloc[0]
                    })
            
            if trial_data:
                trial_df = pd.DataFrame(trial_data)
                normalized_trials.append(trial_df)
            
        except Exception:
            # Logically skip trial on calculation error without halting execution
            skipped_trials += 1
            continue
    
    # Print summary statistics
    print("-" * 40)
    print(f"Preprocessing Summary:")
    print(f"Total Input Trials: {total_trials}")
    print(f"Skipped Trials:     {skipped_trials}")
    print(f"Retained Trials:    {total_trials - skipped_trials}")
    print(f"Retention Rate:     {(total_trials - skipped_trials)/total_trials:.2%}")
    print("-" * 40)
    
    if not normalized_trials:
        raise ValueError("Error: No valid trials remained after filtering.")
    
    result_df = pd.concat(normalized_trials, ignore_index=True)
    numeric_columns = ['normalized_time', 'original_time', 'diameter']
    result_df[numeric_columns] = result_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return result_df

def calculate_quality_metrics(df):
    """
    Calculate post-processing quality metrics.
    """
    quality_metrics = {}
    total_trials_processed = len(df.groupby(['id', 'trialIndex']))
    quality_metrics['retained_trials_count'] = total_trials_processed
    
    # Calculate duration of the analyzed window
    trial_durations = df.groupby(['id', 'trialIndex'])['original_time'].max() - df.groupby(['id', 'trialIndex'])['original_time'].min()
    quality_metrics['mean_trial_duration_ms'] = trial_durations.mean()
    
    return quality_metrics

def plot_average_pupil_response(df, output_path=None):
    """
    Generate and save the average pupil response curve.
    This serves as a visual sanity check for the normalization process.
    """
    plt.figure(figsize=(10, 6))
    
    df['normalized_time'] = pd.to_numeric(df['normalized_time'], errors='coerce')
    df['diameter'] = pd.to_numeric(df['diameter'], errors='coerce')
    df = df.dropna(subset=['normalized_time', 'diameter'])
    
    # Calculate Mean and Standard Error of the Mean (SEM)
    mean_response = df.groupby(['normalized_time', 'Phase'])['diameter'].mean().reset_index()
    std_error = df.groupby(['normalized_time', 'Phase'])['diameter'].sem().reset_index()
    
    # Plot mean line
    sns.lineplot(data=mean_response, x='normalized_time', y='diameter', hue='Phase')
    
    # Plot error bands (SEM)
    for phase in mean_response['Phase'].unique():
        phase_data = mean_response[mean_response['Phase'] == phase]
        phase_std = std_error[std_error['Phase'] == phase]
        
        x_values = phase_data['normalized_time'].values
        y_values = phase_data['diameter'].values
        error_values = phase_std['diameter'].values
        
        plt.fill_between(
            x_values,
            y_values - error_values,
            y_values + error_values,
            alpha=0.2
        )
    
    # Add visual marker for phase transition
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Phase Transition')
    
    plt.title('Average Pupil Response (Time-Normalized)')
    plt.xlabel('Normalized Time')
    plt.ylabel('Pupil Size (mm)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_path:
        save_file = os.path.join(output_path, "pupil_response_normalized_qc.png")
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        print(f"QC Plot saved to: {save_file}")
    
        plt.show() 

def main():
    """Main execution flow."""
    
    # Configuration: file paths (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'demo_raw_data_10subs.jsonl')
    output_dir = os.path.join(script_dir, 'results')
    output_file = os.path.join(output_dir, 'processed_data_normalized.xlsx')
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("1. Loading Data...")
    try:
        df = pd.read_json(file_path, lines=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input Excel file path is correct.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    print("2. Normalizing Trial Time...")
    try:
        df_normalized_time = normalize_trial_time(df)
    except ValueError as e:
        print(f"Processing failed: {e}")
        return
    
    print("3. Saving Processed Data...")
    try:
        df_normalized_time.to_excel(output_file, index=False)
        print(f"Data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return
    
    print("4. Calculating Quality Metrics...")
    quality_metrics = calculate_quality_metrics(df_normalized_time)
    
    # Create a separate sheet for quality metrics
    metrics_df = pd.DataFrame(quality_metrics.items(), columns=['Metric', 'Value'])
    metrics_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
    for metric, value in quality_metrics.items():
        print(f"  - {metric}: {value:.2f}")
    
    print("5. Generating QC Plot...")
    plot_average_pupil_response(df_normalized_time, output_dir)
    
    print("Processing Complete.")

if __name__ == "__main__":
    main()