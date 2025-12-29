import streamlit as st
import pandas as pd
import joblib
import os
import time
from datetime import datetime
from filelock import FileLock

# --- IMPORT PIPELINE FUNCTIONS ---
from utils.preprocessing import preprocess
from drift.drift_detector import detect_drift
from retrain.retrain import retrain

# --- CONFIGURATION ---
DATA_DIR = "data"
MODEL_PATH = "model/current_model.pkl"
REF_DATA_PATH = "data/reference_data.csv"
INCOMING_DATA_PATH = "data/incoming_data.csv"
LOCK_PATH = "data/incoming_data.csv.lock"

st.set_page_config(page_title="ML Drift Pipeline", layout="wide")

# --- SESSION STATE ---
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

def safe_log_data(df):
    """Safely appends data to CSV using a FileLock."""
    lock = FileLock(LOCK_PATH)
    try:
        with lock.acquire(timeout=5):
            if not os.path.exists(INCOMING_DATA_PATH):
                df.to_csv(INCOMING_DATA_PATH, index=False)
            else:
                df.to_csv(INCOMING_DATA_PATH, mode='a', header=False, index=False)
    except TimeoutError:
        st.error("Could not acquire lock to save data.")

def reset_system():
    if os.path.exists(INCOMING_DATA_PATH):
        os.remove(INCOMING_DATA_PATH)
        if os.path.exists(REF_DATA_PATH):
            ref_cols = pd.read_csv(REF_DATA_PATH, nrows=0).columns.tolist()
            pd.DataFrame(columns=ref_cols).to_csv(INCOMING_DATA_PATH, index=False)
    st.session_state.logs = []
    add_log("System reset. Ready.")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙ Control Panel")
    
    st.subheader("Batch Settings")
    batch_size = st.number_input(
        "Batch Size (Drift Check Interval)", 
        min_value=1, 
        value=100, 
        step=10,
        help="How many rows to process before checking for drift."
    )
    
    st.divider()
    if st.button("Reset System Data"):
        reset_system()
        st.rerun()

# --- MAIN UI ---
st.title("🚀 Automated Drift Detection & Retraining")

# Create Tabs
tab_batch, tab_manual = st.tabs(["📂 Batch Stream (CSV)", "✍ Manual Input"])

# ==========================================
# TAB 1: BATCH PROCESSING (The Simulation)
# ==========================================
with tab_batch:
    st.markdown(f"**Status:** Ready to process stream in batches of **{batch_size}**.")
    
    uploaded_file = st.file_uploader("Upload Incoming Data CSV", type=["csv"], key="batch_upload")

    if uploaded_file and st.button("Start Batch Processing"):
        try:
            # --- RESET LOGS AT START OF RUN ---
            st.session_state.logs = [] 
            add_log("--- NEW BATCH PROCESS STARTED ---")
            
            df_stream = pd.read_csv(uploaded_file)
            total_rows = len(df_stream)
            has_labels = "label" in df_stream.columns
            
            add_log(f"File loaded: {total_rows} rows.")
            if has_labels:
                add_log("✅ Labels found: Auto-Retraining ENABLED.")
            else:
                add_log("❌ No Labels: Drift Detection ONLY.")
            
            # UI Elements for Loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_display = st.empty()
            
            # Ensure file exists
            if not os.path.exists(INCOMING_DATA_PATH):
                df_stream.iloc[:0].to_csv(INCOMING_DATA_PATH, index=False)

            # Processing Loop
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = df_stream.iloc[start_idx:end_idx]
                
                # 1. Save Batch
                safe_log_data(batch)
                
                # 2. Check Drift
                status_text.text(f"Processing rows {start_idx} to {end_idx}...")
                drifted_cols = detect_drift(REF_DATA_PATH, INCOMING_DATA_PATH)
                
                if drifted_cols:
                    add_log(f"⚠ Drift Detected in: {drifted_cols}")
                    
                    if has_labels:
                        add_log("🔄 Initiating Auto-Retraining...")
                        result = retrain()
                        
                        if result:
                            if result['status'] == 'replaced':
                                add_log(f"✅ SUCCESS: Model Retrained. New Acc: {result['accuracy']}")
                            elif result['status'] == 'skipped':
                                add_log(f"ℹ Retrain Skipped: {result['reason']}")
                        else:
                            add_log("ℹ Retrain Skipped: Insufficient data.")
                    else:
                        add_log("🛑 Drift ignored: No ground truth.")
                
                # Update UI
                progress_bar.progress(end_idx / total_rows)
                # Show last 10 logs so user sees activity flowing
                log_display.code("\n".join(st.session_state.logs[-10:]), language="text")
                time.sleep(0.1)

            status_text.text("Processing Complete!")
            st.success("Batch processing finished.")
            
        except Exception as e:
            st.error(f"Error: {e}")
            add_log(f"ERROR: {e}")

# ==========================================
# TAB 2: MANUAL INPUT (Single Prediction)
# ==========================================
with tab_manual:
    st.markdown("**Test the model with single manual entries.**")
    
    try:
        model = joblib.load(MODEL_PATH)
    except:
        st.warning("Model not found. Run initial training first.")
        model = None

    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(REF_DATA_PATH):
            ref_df = pd.read_csv(REF_DATA_PATH, nrows=1)
            feature_cols = [c for c in ref_df.columns if c != 'label']
            
            user_input = st.text_input(
                f"Enter {len(feature_cols)} features (comma separated)",
                placeholder="e.g., 0.5, 12.0, 3.5..."
            )
        else:
            st.error("Reference data not found.")
            feature_cols = []
            user_input = ""

    with col2:
        true_label = st.selectbox("Ground Truth Label (Optional)", [None, 0, 1, 2])

    if st.button("Predict & Log"):
        # Reset logs if they are getting too long or if switching context
        if len(st.session_state.logs) > 50:
             st.session_state.logs = []
        
        if model and user_input:
            try:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != len(feature_cols):
                    st.error(f"Expected {len(feature_cols)} values, got {len(values)}.")
                else:
                    input_df = pd.DataFrame([values], columns=feature_cols)
                    X_input = preprocess(input_df)
                    pred = model.predict(X_input)[0]
                    
                    st.info(f"🤖 Model Prediction: **{pred}**")
                    
                    if true_label is not None:
                        input_df['label'] = true_label
                        safe_log_data(input_df)
                        add_log(f"✍ Manual Entry Logged. Label: {true_label}")
                        st.success("Data logged to system.")
                        
                        drifted = detect_drift(REF_DATA_PATH, INCOMING_DATA_PATH)
                        if drifted:
                            add_log(f"⚠ Drift Detected after manual entry.")
                            st.warning("Drift detected! (Run batch process to retrain)")
                    else:
                        add_log("Prediction made (Data not logged - No Label).")
                        
                    st.code("\n".join(st.session_state.logs[-5:]), language="text")

            except Exception as e:
                st.error(f"Error parsing input: {e}")

# --- DOWNLOAD LOGS ---
if st.session_state.logs:
    st.divider()
    st.subheader("📋 Audit Logs")
    full_log = "\n".join(st.session_state.logs)
    st.download_button("Download Log File", full_log, "processing_log.txt", "text/plain")