import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(ref_path, new_path, threshold=0.05):
    # Read the CSV files
    try:
        ref = pd.read_csv(ref_path)
        new = pd.read_csv(new_path)
    except Exception as e:
        print(f"Error reading data for drift detection: {e}")
        return []

    drifted = []

    # DYNAMICALLY identify feature columns (everything except 'label')
    feature_cols = [c for c in ref.columns if c != "label"]

    for col in feature_cols:
        # Check if column exists in new data
        if col not in new.columns:
            continue

        # Force conversion to numeric and drop bad rows
        ref_vals = pd.to_numeric(ref[col], errors='coerce').dropna()
        new_vals = pd.to_numeric(new[col], errors='coerce').dropna()

        # Safety check: Need data to compare
        if len(ref_vals) == 0 or len(new_vals) == 0:
            continue

        # Run KS Test
        stat, p = ks_2samp(ref_vals, new_vals)
        
        if p < threshold:
            drifted.append(col)

    return drifted