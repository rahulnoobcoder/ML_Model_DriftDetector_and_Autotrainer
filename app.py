import streamlit as st
import pandas as pd
import joblib
import os
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from filelock import FileLock
from utils.preprocessing import preprocess
from drift.drift_detector import detect_drift

# --- CONFIGURATION ---
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "current_model.pkl")
REF_DATA_PATH = os.path.join(DATA_DIR, "reference_data.csv")
INCOMING_DATA_PATH = os.path.join(DATA_DIR, "incoming_data.csv")
LOCK_PATH = os.path.join(DATA_DIR, "system.lock")
ADMIN_PASSWORD = "admin123"  # 🔒 Simple Auth Password

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Drift Battle v3.0", layout="wide", page_icon="🛡️")

# --- UTILS ---
if "logs" not in st.session_state: st.session_state.logs = []
if "history" not in st.session_state: st.session_state.history = pd.DataFrame(columns=["Batch", "Accuracy", "Model_Type"])
if "admin_logged_in" not in st.session_state: st.session_state.admin_logged_in = False

def log(msg):
    st.session_state.logs.append(f"{time.strftime('%H:%M:%S')} - {msg}")

def is_system_ready():
    """Checks if base model and ref data exist."""
    return os.path.exists(MODEL_PATH) and os.path.exists(REF_DATA_PATH)

def save_incoming_data(new_df):
    """
    Appends new inputs to the permanent incoming_data.csv log.
    Handles file locking to prevent write conflicts.
    """
    lock = FileLock(f"{INCOMING_DATA_PATH}.lock")
    with lock.acquire(timeout=10):
        # Check if file exists to determine if we need a header
        header = not os.path.exists(INCOMING_DATA_PATH)
        new_df.to_csv(INCOMING_DATA_PATH, mode='a', header=header, index=False)

def train_candidate(X, y):
    model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X, y)
    return model

# --- TABS ---
tab_user, tab_admin = st.tabs(["👤 User Simulation", "🔒 Admin Panel"])

# ==========================================
# 👤 USER SIMULATION (LOCKED IF NOT READY)
# ==========================================
with tab_user:
    st.header("⚔️ Model Drift Simulation")

    # 🛑 1. SYSTEM READINESS CHECK
    if not is_system_ready():
        st.error("🚫 SYSTEM LOCKED")
        st.warning("The AI System has not been initialized yet. Please ask an Administrator to log in and upload the base model.")
        st.stop()  # Stops the rest of the code in this tab from running

    # ... If we pass the check, show the UI ...
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Stream Data (CSV)", type="csv")
    with col2:
        chunk_size = st.number_input("Batch Size", min_value=50, value=500, step=50)

    # PLACEHOLDERS
    st.divider()
    chart_placeholder = st.empty()
    battle_zone = st.container()

    if uploaded_file and st.button("▶️ Start Simulation", type="primary"):
        # Reset Session
        st.session_state.logs = []
        st.session_state.history = pd.DataFrame(columns=["Batch", "Accuracy", "Model_Type"])

        # Load & PERSIST Data
        df_stream = pd.read_csv(uploaded_file)
        
        # 💾 LOGGING: Append to permanent history immediately
        save_incoming_data(df_stream)
        log(f"💾 Data Logged: {len(df_stream)} rows appended to {INCOMING_DATA_PATH}")

        # Load System Resources
        model = joblib.load(MODEL_PATH)
        current_model = model
        
        # Progress Bar
        progress_bar = st.progress(0)
        total_chunks = len(df_stream) // chunk_size

        # --- BATCH LOOP ---
        for i, batch_start in enumerate(range(0, len(df_stream), chunk_size)):
            chunk = df_stream.iloc[batch_start : batch_start + chunk_size].copy()
            if len(chunk) < 50: break 

            progress_bar.progress(min((i + 1) / total_chunks, 1.0))
            
            # Prepare Data
            # Note: We reload ref_df inside loop or before loop to get features
            ref_df = pd.read_csv(REF_DATA_PATH) 
            feature_cols = [c for c in ref_df.columns if c != "label"]
            
            X_chunk = preprocess(chunk[feature_cols])
            y_chunk = chunk['label']
            
            # 1. Performance Check
            curr_acc = accuracy_score(y_chunk, current_model.predict(X_chunk))
            
            # Update Chart
            new_row = pd.DataFrame([{"Batch": i, "Accuracy": curr_acc}])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
            with chart_placeholder:
                st.line_chart(st.session_state.history, x="Batch", y="Accuracy")

            # 2. Drift Check
            temp_path = f"data/temp_batch_{int(time.time())}.csv"
            chunk.to_csv(temp_path, index=False)
            drifted_cols = detect_drift(REF_DATA_PATH, temp_path)
            if os.path.exists(temp_path): os.remove(temp_path)

            if drifted_cols:
                log(f"⚠️ Batch {i}: Drift Detected in {len(drifted_cols)} features")
                
                with battle_zone:
                    # BATTLE LOGIC
                    X_train, X_test, y_train_true, y_test_true = train_test_split(X_chunk, y_chunk, test_size=0.2)
                    
                    # Candidate A (Pseudo)
                    pseudo_labels = current_model.predict(X_train)
                    model_a = train_candidate(X_train, pseudo_labels)
                    acc_a = accuracy_score(y_test_true, model_a.predict(X_test))
                    
                    # Candidate B (True)
                    model_b = train_candidate(X_train, y_train_true)
                    acc_b = accuracy_score(y_test_true, model_b.predict(X_test))
                    
                    if acc_a > acc_b:
                        current_model = model_a
                        log(f"🏆 Batch {i}: Pseudo-Labels Won. Updating Model.")
                    else:
                        current_model = model_b
                        log(f"🏆 Batch {i}: True Labels Won. Updating Model.")

                    # Save winner
                    joblib.dump(current_model, MODEL_PATH)
            else:
                log(f"✅ Batch {i}: Stable.")

        progress_bar.empty()
        st.success("Simulation & Logging Complete")

    # Logs at bottom
    with st.expander("Show Logs"):
        for l in st.session_state.logs: st.text(l)


# ==========================================
# 🔒 ADMIN PANEL (AUTH REQUIRED)
# ==========================================
with tab_admin:
    st.header("🔧 Administrator Panel")

    # 🔐 AUTH CHECK
    if not st.session_state.admin_logged_in:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Incorrect Password")
        st.stop() # Stop rendering admin controls if not logged in

    # 🔓 LOGGED IN VIEW
    st.success("🔓 Authenticated as Admin")
    if st.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()
        
    st.divider()
    st.subheader("System Initialization")
    
    ac1, ac2 = st.columns(2)
    u_csv = ac1.file_uploader("Reference Data (CSV)", type="csv")
    u_pkl = ac2.file_uploader("Base Model (PKL)", type="pkl")

    if st.button("🚀 Initialize / Reset System"):
        if u_csv and u_pkl:
            lock = FileLock(LOCK_PATH)
            with lock.acquire(timeout=10):
                # Save Ref
                df = pd.read_csv(u_csv)
                df.to_csv(REF_DATA_PATH, index=False)
                # Save Model
                with open(MODEL_PATH, "wb") as f: f.write(u_pkl.getbuffer())
                # Reset Preprocessor
                preprocess(df.drop(columns=['label'], errors='ignore'), training=True)
                # Clear old logs if resetting
                if os.path.exists(INCOMING_DATA_PATH): os.remove(INCOMING_DATA_PATH)
                
                st.success("System Initialized! User Tab Unlocked.")
        else:
            st.error("Please upload both files.")

    # Show Data Log Stats
    if os.path.exists(INCOMING_DATA_PATH):
        st.divider()
        st.subheader("📊 Global Incoming Data Log")
        hist_df = pd.read_csv(INCOMING_DATA_PATH)
        st.write(f"Total Records Logged: **{len(hist_df)}**")
        with st.expander("View Last 5 Rows"):
            st.dataframe(hist_df.tail())