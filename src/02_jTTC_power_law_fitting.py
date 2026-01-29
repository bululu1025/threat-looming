"""
Project: Hanmengfan Looming Fear - Power Law Fitting
Target Journal: PLOS Biology
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
import os

# Set warnings to ignore
warnings.filterwarnings('ignore')

# 1. Data Loading and Preprocessing
# ==========================================
# Data path: relative to this script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "demo_data.xlsx")
try:
    df = pd.read_excel(file_path, sheet_name="jTTC")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
    exit()

# Variable Mapping
treatment_names = {1: 'LT', 2: 'PLC', 3: 'AVP'}
df['treatment_name'] = df['treatment'].map(treatment_names)
df['threat_name'] = df['Isthreaten'].map({1: 'Nonthreatening', 2: 'Threatening'})
df['gender_name'] = df['gender'].map({1: 'Male', 2: 'Female'})

print(f"Data Loaded. Shape: {df.shape}")

# 2. Model Definition (compare with linear model)
# ==========================================
def power_law(x, alpha, beta):
    """Power Law Model: jTTC = alpha * (aTTC)^beta"""
    return alpha * (x ** beta)

def linear_model(x, alpha, beta):
    """Linear Model (Baseline): jTTC = alpha + beta * aTTC"""
    return alpha + beta * x

# 3. Model Fitting (for each subject and condition)
# ==========================================
results = []
formal_list = df['Formal'].unique()

print("Starting model fitting for each subject and condition...")

for formal in formal_list:
    subject_data = df[df['Formal'] == formal]
    
    for threat_level in [1, 2]:
        condition_data = subject_data[subject_data['Isthreaten'] == threat_level]
        
        # Exclude cases with insufficient data points
        if len(condition_data) < 5:
            continue
            
        # Extract metadata
        treatment = condition_data['treatment'].iloc[0]
        gender = condition_data['gender'].iloc[0]
        
        # Extract independent variable (aTTC) and dependent variable (jTTC/RT)
        aTTC = condition_data['TTC'].values
        jTTC = condition_data['RT'].values
        
        try:
            # A. Fit power law model (Power Law)
            # Set boundaries to prevent parameter extremes: alpha [0.1, 10], beta [0.1, 2]
            popt_power, _ = curve_fit(power_law, aTTC, jTTC, 
                                      bounds=([0.1, 0.1], [10, 2]), maxfev=5000)
            alpha_power, beta_power = popt_power
            
            # Calculate goodness of fit R^2
            power_predictions = power_law(aTTC, alpha_power, beta_power)
            ss_res_power = np.sum((jTTC - power_predictions) ** 2)
            ss_tot = np.sum((jTTC - np.mean(jTTC)) ** 2)
            r2_power = 1 - (ss_res_power / ss_tot)
            power_rmse = np.sqrt(np.mean((jTTC - power_predictions) ** 2))
            
            # B. Fit linear model (Linear Baseline) - for comparison
            popt_linear, _ = curve_fit(linear_model, aTTC, jTTC,
                                       bounds=([-10, -2], [10, 2]), maxfev=5000)
            alpha_linear, beta_linear = popt_linear
            
            linear_predictions = linear_model(aTTC, alpha_linear, beta_linear)
            ss_res_linear = np.sum((jTTC - linear_predictions) ** 2)
            r2_linear = 1 - (ss_res_linear / ss_tot)
            linear_rmse = np.sqrt(np.mean((jTTC - linear_predictions) ** 2))
            
            # Save results
            results.append({
                'Formal': formal,
                'treatment_name': treatment_names[treatment],
                'threat_name': 'Threatening' if threat_level == 2 else 'Nonthreatening',
                'gender_name': 'Female' if gender == 2 else 'Male',
                # Power Law Parameters
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'r2_power': r2_power,
                'rmse_power': power_rmse,
                # Linear Model Parameters (for comparison)
                'r2_linear': r2_linear,
                'rmse_linear': linear_rmse,
                'delta_r2': r2_power - r2_linear # Positive means Power Law is better
            })
            
        except Exception as e:
            # Record cases with fitting failure
            continue

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"Model fitting complete. Successfully fitted {len(results_df)} conditions.")

# Export fitting parameter results
results_df.to_csv('model_fitting_parameters.csv', index=False)
print("Parameters saved to 'model_fitting_parameters.csv'.")

# 4. Statistical Analysis
# ==========================================
print("\n=== Statistical Analysis: Linear Mixed Effects Models ===")

# 4.1 Compare model goodness of fit (Model Comparison)
mean_r2_power = results_df['r2_power'].mean()
mean_r2_linear = results_df['r2_linear'].mean()
better_fit_count = sum(results_df['r2_power'] > results_df['r2_linear'])
print(f"Mean R2 (Power Law): {mean_r2_power:.4f}")
print(f"Mean R2 (Linear): {mean_r2_linear:.4f}")
print(f"Power Law provided better fit in {better_fit_count}/{len(results_df)} cases.")

# 4.2 Mixed effects model analysis for power law exponent Beta
# Analyze the effect of Treatment, Threat, Gender on time compression effect (Beta)
print("\n--- LMM Results for Exponent Beta (Time Compression) ---")
beta_formula = '''beta_power ~ C(treatment_name, Treatment("PLC")) * 
                               C(threat_name, Treatment("Nonthreatening")) * 
                               C(gender_name, Treatment("Male"))'''

# Use 'Formal' (subject ID) as random intercept
try:
    beta_model = mixedlm(beta_formula, results_df, groups=results_df['Formal']).fit()
    print(beta_model.summary())
except Exception as e:
    print(f"LMM for Beta failed to converge: {e}")

# 4.3 Mixed effects model analysis for power law coefficient Alpha
# Analyze the effect of Treatment, Threat, Gender on baseline response (Alpha)
print("\n--- LMM Results for Coefficient Alpha (Baseline Scaling) ---")
alpha_formula = '''alpha_power ~ C(treatment_name, Treatment("PLC")) * 
                                C(threat_name, Treatment("Nonthreatening")) * 
                                C(gender_name, Treatment("Male"))'''

try:
    alpha_model = mixedlm(alpha_formula, results_df, groups=results_df['Formal']).fit()
    print(alpha_model.summary())
except Exception as e:
    print(f"LMM for Alpha failed to converge: {e}")

print("\nAnalysis script finished.")