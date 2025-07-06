# app.py
import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💸 Loan Default Prediction App")
st.markdown("Enter the loan details below to predict if the borrower will **default** or **pay** the loan.")

# --- INPUT FORM ---
with st.form("loan_form"):
    st.subheader("📋 Loan Details")
    
    out_prncp = st.number_input("Outstanding Principal", min_value=0.0, step=100.0, format="%.2f")
    out_prncp_inv = st.number_input("Outstanding Principal (Investor)", min_value=0.0, step=100.0, format="%.2f")
    last_pymnt_amnt = st.number_input("Last Payment Amount", min_value=0.0, step=100.0, format="%.2f")
    total_rec_prncp = st.number_input("Total Received Principal", min_value=0.0, step=100.0, format="%.2f")
    recoveries = st.number_input("Recoveries", min_value=0.0, step=10.0, format="%.2f")
    collection_recovery_fee = st.number_input("Collection Recovery Fee", min_value=0.0, step=10.0, format="%.2f")
    
    st.subheader("📊 Credit Score Range")
    last_fico_range_low = st.slider("Last FICO Score (Low)", min_value=300, max_value=850, value=650)
    last_fico_range_high = st.slider("Last FICO Score (High)", min_value=300, max_value=850, value=680)
    
    total_pymnt = st.number_input("Total Payment", min_value=0.0, step=100.0, format="%.2f")

    # Submit button
    submit = st.form_submit_button("🔍 Predict")

# --- PREDICTION ---
if submit:
    input_data = np.array([[out_prncp, out_prncp_inv, last_pymnt_amnt, total_rec_prncp, recoveries,
                            collection_recovery_fee, last_fico_range_low, last_fico_range_high, total_pymnt]])
    
    prediction = model.predict(input_data)[0]
    label = "✅ Paid" if prediction == 0 else "❌ Defaulted"

    st.success(f"### Prediction: **{label}**")
