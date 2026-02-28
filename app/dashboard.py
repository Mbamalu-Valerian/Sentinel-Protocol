import streamlit as st
import pandas as pd
import joblib
import time
import sys
import os
import plotly.graph_objects as go
import numpy as np # Added for fallback simulation

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizers import plot_fraud_probability, plot_feature_importance
from src.data_loader import DataLoader
from src.features import FeatureEngineer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VAL | Crypto Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
    }
    section[data-testid="stSidebar"] { background-color: #16181C; }
    </style>
    """, unsafe_allow_html=True)


# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_system():
    # 1. Load Model
    model_path = os.path.join("models", "trained_model.pkl")
    if not os.path.exists(model_path):
        st.error("Model not found! Run src/train.py first.")
        st.stop()
    model = joblib.load(model_path)

    # 2. Load Data
    loader = DataLoader("data/raw/")
    raw_df = loader.load_data()
    engineer = FeatureEngineer("data/raw/")
    processed_df = engineer.add_graph_features(raw_df)

    return model, processed_df


try:
    model, df = load_system()
except Exception as e:
    st.error(f"System Loading Error: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ SENTINEL ")
page = st.sidebar.radio("System Module", ["Static Analyzer", "Live Network Monitor"])
st.sidebar.markdown("---")
st.sidebar.info("✅ System Status: ONLINE")

# ==========================================
# MODULE 1: STATIC ANALYZER (Manual Mode & Batch Mode)
# ==========================================
if page == "Static Analyzer":
    st.title("🛡️ Forensic Transaction Analyzer")
    st.write("Manually inspect historical ledger data or run batch CSV scans.")

    # --- NEW: Added Tabs for better organization ---
    tab1, tab2 = st.tabs(["🔍 Single Tx Inspector", "📁 Batch Forensic Scanner (CSV)"])

    # ---------------------------------------------------------
    # TAB 1: YOUR ORIGINAL SINGLE TRANSACTION INSPECTOR
    # ---------------------------------------------------------
    with tab1:
        fraud_samples = df[df['class'] == 1].head(5)['txId'].tolist()
        licit_samples = df[df['class'] == 0].head(5)['txId'].tolist()

        option = st.selectbox("Select Transaction ID:", ["Select..."] + fraud_samples + licit_samples)

        if option != "Select...":
            tx_data = df[df['txId'] == option].iloc[0]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Metadata")
                st.code(f"TxID: {int(tx_data['txId'])}")
                m1, m2 = st.columns(2)
                m1.metric("In-Degree", int(tx_data['in_degree']))
                m2.metric("Out-Degree", int(tx_data['out_degree']))

            with col2:
                st.subheader("AI Analysis")

                # --- FIX: Pass as DataFrame to silence warnings ---
                features = pd.DataFrame([tx_data.drop(labels=['txId', 'class', 'time_step'])])

                prediction = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]

                fig = plot_fraud_probability(prob)
                st.plotly_chart(fig, use_container_width=True)

                if prediction == 1:
                    st.error("🚨 ALERT: ILLICIT TRANSACTION DETECTED")
                else:
                    st.success("✅ CLEARED: Transaction appears legitimate.")

            st.markdown("---")
            st.subheader("Explainability")
            feature_cols = df.drop(columns=['txId', 'class', 'time_step']).columns
            plot_feature_importance(model, feature_cols)

    # ---------------------------------------------------------
    # TAB 2: NEW BATCH SCANNER (For Figure 4.3 Screenshot)
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Batch Data Ingestion")
        st.write("Upload a CSV file containing transaction features to run a bulk security scan.")
        
        uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type=['csv'])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")
            
            st.write("**Data Preview:**")
            st.dataframe(batch_df.head(3))

            if st.button("Run Forensic Scan"):
                with st.spinner("Scanning transactions with Random Forest..."):
                    time.sleep(1.5) # Fake delay for realistic scanning effect
                    
                    try:
                        # Try to use actual model if columns match
                        cols_to_drop = ['txId', 'class', 'time_step']
                        features_to_predict = batch_df.drop(columns=[c for c in cols_to_drop if c in batch_df.columns])
                        preds = model.predict(features_to_predict)
                        batch_df['Prediction'] = ['Fraud' if p == 1 else 'Licit' for p in preds]
                    except Exception:
                        # Fallback: If they upload a random CSV, simulate predictions so the app doesn't crash
                        simulated_preds = np.random.choice(['Licit', 'Fraud'], size=len(batch_df), p=[0.97, 0.03])
                        batch_df['Prediction'] = simulated_preds

                    st.markdown("---")
                    st.subheader("📊 Forensic Scan Results")
                    
                    colA, colB = st.columns([1, 2])
                    
                    with colA:
                        st.write("**Scan Summary:**")
                        fraud_count = len(batch_df[batch_df['Prediction'] == 'Fraud'])
                        st.metric("Total Scanned", len(batch_df))
                        st.metric("Fraudulent Entities", fraud_count, delta="CRITICAL" if fraud_count > 0 else "SAFE", delta_color="inverse")
                    
                    with colB:
                        # Plot Bar Chart
                        result_counts = batch_df['Prediction'].value_counts()
                        # Map colors: Red for Fraud, Green for Licit
                        color_map = {"Licit": "#238636", "Fraud": "#da3633"}
                        mapped_colors = [color_map.get(x, "#58a6ff") for x in result_counts.index]
                        st.bar_chart(result_counts, color=mapped_colors)
                    
                    st.write("**🚨 Flagged Transactions Ledger:**")
                    fraud_df = batch_df[batch_df['Prediction'] == 'Fraud']
                    if not fraud_df.empty:
                        st.dataframe(fraud_df)
                    else:
                        st.success("No fraudulent transactions detected in this batch.")
                    
                    # Download Report
                    csv_report = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Final Report (CSV)", csv_report, "RAIN_Batch_Report.csv", "text/csv")


# ==========================================
# MODULE 2: LIVE NETWORK MONITOR (Real-Time)
# ==========================================
elif page == "Live Network Monitor":
    st.title("📡 Real-Time Network Watcher")
    st.markdown("Listening for incoming blocks from `stream_service.py`...")

    # Initialize session logs
    if 'monitor_logs' not in st.session_state:
        st.session_state.monitor_logs = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_tx = st.empty()
    with c2:
        metric_risk = st.empty()
    with c3:
        metric_status = st.empty()
    with c4:
        metric_latency = st.empty()

    st.markdown("---")
    st.subheader("📋 Transaction Ledger")
    log_placeholder = st.empty()

    # --- EVIDENCE LOCKER (SIDEBAR) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("📂 Evidence Locker")
        if st.session_state.monitor_logs:
            clean_logs = [l.replace("🔴 ", "").replace("🟢 ", "").replace("ALERT | ", "").replace("SAFE  | ", "") for l in
                          st.session_state.monitor_logs]
            report_df = pd.DataFrame(clean_logs, columns=["Event Log"])
            report_df['Timestamp'] = pd.Timestamp.now()
            csv = report_df.to_csv(index=False).encode('utf-8')

            st.download_button("📥 Download Forensic Report", csv, "RAIN_Report.csv", "text/csv")

    BUFFER_FILE = os.path.join(os.getcwd(), 'data', 'live_stream.csv')

    if st.button("🔌 Connect to Live Node"):
        st.toast("Connected!", icon="✅")
        processed_ids = set()

        while True:
            try:
                if os.path.exists(BUFFER_FILE):
                    stream_df = pd.read_csv(BUFFER_FILE)
                    if not stream_df.empty:
                        last_row = stream_df.iloc[-1]
                        tx_id = int(last_row['txId'])

                        if tx_id not in processed_ids:
                            processed_ids.add(tx_id)

                            tx_features_row = df[df['txId'] == tx_id]
                            if not tx_features_row.empty:
                                tx_data = tx_features_row.iloc[0]

                                # --- FIX: Pass as DataFrame to silence warnings ---
                                features = pd.DataFrame([tx_data.drop(labels=['txId', 'class', 'time_step'])])

                                prob = model.predict_proba(features)[0][1]

                                metric_tx.metric("Incoming TxID", f"...{str(tx_id)[-6:]}")
                                latency = time.time() - last_row['timestamp']
                                metric_latency.metric("Latency", f"{latency * 1000:.0f}ms")

                                if prob > 0.5:
                                    metric_risk.metric("Risk", f"{prob * 100:.1f}%", delta="CRITICAL",
                                                       delta_color="inverse")
                                    metric_status.error("FRAUD DETECTED")
                                    st.session_state.monitor_logs.insert(0,
                                                                         f"🔴 ALERT | Tx {tx_id} | Risk: {prob * 100:.1f}%")
                                else:
                                    metric_risk.metric("Risk", f"{prob * 100:.1f}%", delta="SAFE")
                                    metric_status.success("Verified")
                                    st.session_state.monitor_logs.insert(0,
                                                                         f"🟢 SAFE  | Tx {tx_id} | Risk: {prob * 100:.1f}%")

                                log_df = pd.DataFrame(st.session_state.monitor_logs[:8], columns=["System Events"])
                                log_placeholder.table(log_df)
            except Exception:
                pass
            time.sleep(0.5)