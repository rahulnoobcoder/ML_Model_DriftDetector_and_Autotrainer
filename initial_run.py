import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from utils.preprocessing import preprocess

# --- CONFIGURATION ---
FEATURE_COLS = ["Annual_Income", "Credit_Score", "Loan_Amount"]
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/current_model.pkl"

def generate_and_save_data():
    print("🏦 Generating Loan Data (3 Features)...")
    
    # 1. Generate Data
    n_samples = 10000
    np.random.seed(42)

    data = {
        "Annual_Income": np.random.randint(25000, 150000, n_samples),
        "Credit_Score": np.random.randint(500, 850, n_samples),
        "Loan_Amount": np.random.randint(5000, 50000, n_samples),
    }
    df = pd.DataFrame(data)

    # 2. Logic: Score > 650 AND Income > 2.5x Loan = Approved
    df["label"] = ((df["Credit_Score"] > 650) & 
                   (df["Annual_Income"] > (df["Loan_Amount"] * 2.5))).astype(int)

    # 3. Split: Ref (Clean) vs Stream (Drifting)
    df_ref = df.iloc[:1000].copy()
    df_stream = df.iloc[1000:].copy()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Files
    df_ref.to_csv(f"{DATA_DIR}/reference_data.csv", index=False)
    
    # Inject Drift into Stream (Score drops over time)
    print("📉 Injecting progressive drift...")
    df_stream.reset_index(drop=True, inplace=True)
    df_stream.loc[2000:4000, "Credit_Score"] -= 50
    df_stream.loc[4000:6000, "Credit_Score"] -= 100
    df_stream.loc[6000:, "Credit_Score"] -= 150

    df_stream.to_csv(f"{DATA_DIR}/drifted_stream.csv", index=False)
    
    # Reset Log
    if os.path.exists(f"{DATA_DIR}/incoming_data.csv"):
        os.remove(f"{DATA_DIR}/incoming_data.csv")
    
    print("✅ Data Generated.")
    return df_ref

def train_base_model(df_ref):
    print("🧠 Training Base Model...")
    X = preprocess(df_ref[FEATURE_COLS], training=True)
    y = df_ref['label']
    
    model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    print("✅ Model Saved.")

if __name__ == "__main__":
    ref = generate_and_save_data()
    train_base_model(ref)