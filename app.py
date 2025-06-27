# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Step 1: Load model and column names safely
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load('loan_pipeline.pkl')
        training_columns = joblib.load('training_columns.pkl')
        return pipeline, training_columns
    except Exception as e:
        st.error(f"❌ Failed to load model or columns: {e}")
        return None, None

pipeline, training_columns = load_artifacts()

st.title("💰 Loan Default Prediction App")

if pipeline is None:
    st.stop()

# Step 2: Create user input fields dynamically
def user_input_form():
    st.sidebar.header("📋 Enter Applicant Information")

    user_data = {}
    for col in training_columns:
        if col.startswith("int_") or col.startswith("num_") or "amount" in col.lower():
            user_data[col] = st.sidebar.number_input(f"{col}", min_value=0.0, step=100.0)
        elif col.endswith("_year") or "term" in col.lower():
            user_data[col] = st.sidebar.number_input(f"{col}", min_value=0, step=1)
        else:
            user_data[col] = st.sidebar.text_input(f"{col}")

    return pd.DataFrame([user_data])

# Step 3: Get input and make prediction
input_df = user_input_form()

if st.button("🔍 Predict Loan Status"):
    try:
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Prediction: Fully Paid with {proba*100:.2f}% confidence.")
        else:
            st.error(f"⚠️ Prediction: Charged Off with {(1-proba)*100:.2f}% confidence.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
