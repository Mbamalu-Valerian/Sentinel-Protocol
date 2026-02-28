import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Fraud Sentinel",
    page_icon="🛡️",
    layout="wide"
)


# --- Load Resources (Cached for Speed) ---
@st.cache_resource
def load_model_and_data():
    base_path = os.getcwd()
    model_path = os.path.join(base_path, 'models', 'random_forest_model.pkl')
    data_path = os.path.join(base_path, 'data', 'processed', 'train_ready_data.csv')

    # Load Model
    model = joblib.load(model_path)

    # Load a sample of data (First 500 rows for demo speed)
    df = pd.read_csv(data_path).sample(n=500, random_state=42)
    return model, df


try:
    model, df_sample = load_model_and_data()
except FileNotFoundError:
    st.error("Error: Could not find model or data. Did you run train_model.py?")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🛡️ Control Panel")
st.sidebar.info("Select a transaction ID from the historical ledger to analyze.")

# Filter to get some Fraud and Safe examples for the demo
fraud_txs = df_sample[df_sample['is_fraud'] == 1]['txId'].head(10).tolist()
safe_txs = df_sample[df_sample['is_fraud'] == 0]['txId'].head(10).tolist()

# Dropdown to select transaction
tx_options = ["--- Select ID ---"] + fraud_txs + safe_txs
selected_tx_id = st.sidebar.selectbox("Select Transaction ID:", tx_options)

# --- Main Dashboard ---
st.title("Blockchain Fraud Detection System")
st.markdown("""
This system uses a **Random Forest Classifier** to detect illicit transaction patterns in real-time.
It analyzes **temporal behaviors** and **network graph metrics**.
""")

if selected_tx_id != "--- Select ID ---":
    # Get the row for this ID
    tx_data = df_sample[df_sample['txId'] == selected_tx_id]

    # Prepare data for prediction (Drop ID, Class, Target)
    # We must ensure columns match exactly what the model was trained on
    X_input = tx_data.drop(columns=['txId', 'class', 'is_fraud'])

    # --- layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Transaction Details")
        st.write(f"**Transaction ID:** {selected_tx_id}")
        st.write(f"**Time Step:** {tx_data['time_step'].values[0]}")
        st.write(f"**In-Degree (Inputs):** {tx_data['in_degree'].values[0]}")
        st.write(f"**Out-Degree (Outputs):** {tx_data['out_degree'].values[0]}")

    with col2:
        st.subheader("🤖 AI Analysis")
        if st.button("Run Fraud Scan"):
            # Prediction
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]  # Probability of being Fraud (1)

            # Display Result
            if prediction == 1:
                st.error("🚨 ALERT: ILLICIT TRANSACTION DETECTED")
                st.metric("Fraud Probability", f"{probability:.2%}")
                st.write("This transaction exhibits high-risk behavioral patterns.")
            else:
                st.success("✅ CLEARED: LICIT TRANSACTION")
                st.metric("Safety Score", f"{1 - probability:.2%}")
                st.write("This transaction appears normal.")

    # --- Feature Importance Visualization ---
    st.markdown("---")
    st.subheader("📊 Model Explainability")
    st.write("Which features contributed most to this decision?")

    # Get importance from model
    importances = model.feature_importances_
    feature_names = X_input.columns

    # Create a DataFrame for plotting
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
    ax.set_title("Top 10 Risk Factors")
    st.pyplot(fig)

else:
    st.info("👈 Please select a Transaction ID from the sidebar to begin.")