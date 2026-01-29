import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

def load_and_prepare_data(file_path, sheet_name=None):
    """
    load and clean data
    """
    print(f"Reading data: {file_path}")
    
    # Determine file type and read accordingly
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(file_path)
    
    # Basic cleaning
    df = df[df['diameter'] > 0].copy() # 确保是副本
    df['normalized_time'] = pd.to_numeric(df['normalized_time'])
    df['diameter'] = pd.to_numeric(df['diameter'])
    
    # Convert categorical variables (Categorical Variables)
    # Ensure Reference Group (Reference Group) is set correctly
    df['treatment'] = pd.Categorical(df['treatment'], categories=[2, 1, 3], ordered=True) # 2为对照
    df['PSV'] = pd.Categorical(df['PSV'], categories=[3, 1, 2, 4, 5], ordered=True)       # 3为对照
    df['gender'] = df['gender'].astype('category')
    df['Isthreaten'] = df['Isthreaten'].astype('category')
    
    print(f"Data loaded. Sample size: {len(df)}")
    return df

def run_full_interaction_flmm(df):
    """
    Run FLMM model with main effects, second-order and third-order interactions
    """
    print("\n" + "="*60)
    print("Building FLMM model (B-Spline Basis)")
    print("Contains: main effects + second-order interactions + third-order interactions")
    print("="*60)

    # --- 1. Define degrees of freedom for B-Spline ---
    # df=4 usually corresponds to a cubic B-Spline with one internal node.
    # This is enough to fit the "shrink-recovery" curve shape of most pupil responses.
    # If the model does not converge, try reducing to df=3.
    spline_df = 4
    
    # --- 2. Build formula ---
    # Use patsy's '** 3' syntax.
    # (A + B + C) ** 3 means: A, B, C, A:B, A:C, B:C, A:B:C
    # Here we treat 'bs(normalized_time)' as one of the factors.
    
    formula = (
        'diameter ~ '
        '(' 
        f'bs(normalized_time, df={spline_df}) + ' # time-based spline factor
        'treatment + '
        'Isthreaten + '
        'PSV + '
        'gender'
        ') ** 3' # Request highest 3-order interactions
    )
    
    print(f"Used formula syntax: {formula}")
    print("Note: Since it contains third-order interactions and spline functions, the parameter number will be very large, and the calculation may be slow...")

    # --- 3. Define random effects ---
    # Ideally, it is random spline, but the calculation is too large.
    # Here we use "random intercept + random linear slope", allowing each person's baseline and overall linear trend to be different.
    re_formula = '~normalized_time'

    # --- 4. Build model ---
    model = sm.MixedLM.from_formula(
        formula,
        groups='id',
        data=df,
        re_formula=re_formula
    )
    
    try:
        # method='lbfgs' has less memory usage, suitable for large parameter models
        # maxiter increase iteration times to ensure convergence
        result = model.fit(method='lbfgs', maxiter=2000)
        
        print("\nModel fitting successful!")
        
        # --- 5. Output results (Summary) ---
        # print(result.summary()) 
        
        # --- 6. Perform Wald test (Type III ANOVA style) ---
        # It will test whether a group of coefficients (e.g., all spline coefficients related to treatment:time) are jointly 0.
        print("\n" + "-"*30)
        print("Wald Tests (Joint Significance Test)")
        print("-"*30)
        print("Explanation:")
        print("1. Find items containing 'bs(normalized_time)'.")
        print("2. If 'bs(...):treatment' is significant, it means that Treatment has changed the 'curve shape' of the pupil response.")
        print("-"*30)
        
        # wald_test_terms will automatically pack all spline coefficients belonging to the same interaction item for testing
        wald_table = result.wald_test_terms()
        
        # 格式化输出，按 P 值排序或保持原样
        print(wald_table.summary_frame().round(4))
        
        return result, wald_table

    except np.linalg.LinAlgError:
        print("Error: Singular Matrix. The model is too complex, and there is multicollinearity between variables.")
        print("Suggestion: Reduce the order of interactions, or reduce the classification levels of PSV/Treatment.")
        return None, None
    except Exception as e:
        print(f"Model fitting failed: {str(e)}")
        return None, None

def main():
    # File path: relative to this script directory (demo data with 10 subjects)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'demo_data.xlsx')
    sheet_name = 'pupil'
    
    # 1. Load data
    df = load_and_prepare_data(file_path, sheet_name=sheet_name)
    
    # 2. Run FLMM
    result, wald_table = run_full_interaction_flmm(df)
    
    # 3. If successful, save Wald table to a file next to the script
    if wald_table is not None:
        output_path = os.path.join(script_dir, 'flmm_wald_results.xlsx')
        wald_table.summary_frame().to_csv(output_path)
        print(f"\nWald test results have been saved to: {output_path}")

if __name__ == "__main__":
    main()