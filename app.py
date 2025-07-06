import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_model.pkl")

st.title("Loan Default Prediction")

# Minimal Inputs
out_prncp = st.number_input("Outstanding Principal", min_value=0)
out_prncp_inv = st.number_input("Outstanding Principal (Investor)", min_value=0)
last_pymnt_amnt = st.number_input("Last Payment Amount", min_value=0)
total_rec_prncp = st.number_input("Total Received Principal", min_value=0)
recoveries = st.number_input("Recoveries", min_value=0)
collection_recovery_fee = st.number_input("Collection Recovery Fee", min_value=0)
last_fico_range_low = st.number_input("Last FICO Range Low", min_value=300, max_value=850)
last_fico_range_high = st.number_input("Last FICO Range High", min_value=300, max_value=850)
total_pymnt = st.number_input("Total Payment", min_value=0)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([[
        out_prncp_inv, out_prncp, last_pymnt_amnt, total_rec_prncp,
        recoveries, collection_recovery_fee, last_fico_range_low,
        last_fico_range_high, total_pymnt
    ]], columns=[
        'out_prncp_inv', 'out_prncp', 'last_pymnt_amnt', 'total_rec_prncp',
        'recoveries', 'collection_recovery_fee', 'last_fico_range_low',
        'last_fico_range_high', 'total_pymnt'
    ])
    
    prediction = model.predict(input_df)[0]
    result = "Default" if prediction == 1 else "Paid"
    st.success(f"Prediction: {result}")
