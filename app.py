import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model (upload loan_default_model.pkl to repo)
with open('loan_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Loan Default Prediction")

st.write("""
Enter the loan details below to predict if the loan is likely to default.
""")

# Input fields for the 9 features
out_prncp_inv = st.number_input("Outstanding Principal Inv", min_value=0.0, format="%.2f")
out_prncp = st.number_input("Outstanding Principal", min_value=0.0, format="%.2f")
last_pymnt_amnt = st.number_input("Last Payment Amount", min_value=0.0, format="%.2f")
total_rec_prncp = st.number_input("Total Principal Received", min_value=0.0, format="%.2f")
recoveries = st.number_input("Recoveries", min_value=0.0, format="%.2f")
collection_recovery_fee = st.number_input("Collection Recovery Fee", min_value=0.0, format="%.2f")
last_fico_range_low = st.number_input("Last FICO Range Low", min_value=300, max_value=850, step=1)
last_fico_range_high = st.number_input("Last FICO Range High", min_value=300, max_value=850, step=1)
total_pymnt = st.number_input("Total Payment Received", min_value=0.0, format="%.2f")

# Prepare input data for prediction
input_data = np.array([[
    out_prncp_inv, out_prncp, last_pymnt_amnt, total_rec_prncp,
    recoveries, collection_recovery_fee, last_fico_range_low,
    last_fico_range_high, total_pymnt
]])

# Button for prediction
if st.button("Predict Default Risk"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Loan is likely to DEFAULT with probability {proba:.2f}")
    else:
        st.success(f"✅ Loan is likely to be REPAID with probability {1 - proba:.2f}")
