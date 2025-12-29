import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

ENCODER_PATH = "model/encoders.pkl"

def load_encoders():
    if os.path.exists(ENCODER_PATH):
        return joblib.load(ENCODER_PATH)
    return {}

def save_encoders(encoders):
    # Ensure directory exists
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)

def preprocess(df, training=False):
    df = df.copy()
    encoders = load_encoders()
    
    # Track if we updated any encoders to save them later
    updated = False

    for col in df.columns:
        if df[col].dtype == "object":
            if training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
                updated = True
            else:
                # During inference, use the saved encoder
                le = encoders.get(col)
                if le:
                    # Handle unseen labels safely (optional: map to unknown or mode)
                    # For now, we try/except to avoid crashing on new data
                    try:
                        df[col] = le.transform(df[col])
                    except ValueError:
                        # Fallback for unseen labels (e.g., set to -1 or most frequent)
                        df[col] = -1 
                else:
                    # If no encoder exists for this col, we can't process it.
                    pass

    if updated and training:
        save_encoders(encoders)

    return df