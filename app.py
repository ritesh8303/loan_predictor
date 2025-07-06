import streamlit as st
import pandas as pd
import joblib
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Assets ---
# Load the pre-trained model pipeline
try:
    model = joblib.load('loan_default_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the training script first to generate 'loan_default_model.pkl'.")
    st.stop()

# Load the feature columns
try:
    with open('columns.json', 'r') as f:
        model_columns = json.load(f)
except FileNotFoundError:
    st.error("Columns file not found. Please run the training script first to generate 'columns.json'.")
    st.stop()

# Load the grade values for the dropdown
try:
    with open('grade_values.json', 'r') as f:
        grade_values = json.load(f)
except FileNotFoundError:
    st.error("Grade values file not found. Please run the training script first to generate 'grade_values.json'.")
    st.stop()


# --- Application UI ---
st.title("Loan Default Prediction App")
st.markdown("""
This app predicts the likelihood of a loan defaulting based on a few key features.
Please enter the applicant's details below.
""")

# --- Sidebar with Input Fields ---
st.sidebar.header("Applicant Information")

def user_input_features():
    """Creates sidebar input fields and returns a DataFrame."""
    loan_amnt = st.sidebar.number_input('Loan Amount ($)', min_value=500, max_value=40000, value=10000, step=500)
    int_rate = st.sidebar.slider('Interest Rate (%)', min_value=5.0, max_value=31.0, value=12.0, step=0.1)
    grade = st.sidebar.selectbox('Loan Grade', options=grade_values, index=grade_values.index('B'))
    annual_inc = st.sidebar.number_input('Annual Income ($)', min_value=10000, max_value=10000000, value=75000, step=1000)
    dti = st.sidebar.slider('Debt-to-Income Ratio (DTI)', min_value=0.0, max_value=50.0, value=15.0, step=0.1)

    data = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'grade': grade,
        'annual_inc': annual_inc,
        'dti': dti
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Panel for Displaying Inputs and Prediction ---
st.subheader("Summary of Applicant's Details")

# Display the user inputs in a clean format
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Loan Amount:** ${input_df['loan_amnt'].iloc[0]:,}")
    st.info(f"**Interest Rate:** {input_df['int_rate'].iloc[0]}%")
    st.info(f"**Loan Grade:** {input_df['grade'].iloc[0]}")
with col2:
    st.info(f"**Annual Income:** ${input_df['annual_inc'].iloc[0]:,}")
    st.info(f"**DTI Ratio:** {input_df['dti'].iloc[0]}")


# Ensure the input dataframe has columns in the same order as the model expects
input_df = input_df[model_columns]

# --- Prediction Logic ---
if st.button('Predict Loan Status', key='predict_button'):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        # Get prediction probability
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result')
        
        # Display the prediction
        if prediction[0] == 1:
            st.error('**Status: Loan is likely to DEFAULT**', icon="⚠️")
        else:
            st.success('**Status: Loan is likely to be PAID**', icon="✅")

        # Display the confidence score
        confidence = prediction_proba[0][prediction[0]]
        st.metric(label="Prediction Confidence", value=f"{confidence:.2%}")

        # Explanation of confidence
        st.markdown(f"""
        This confidence score represents the model's certainty in its prediction.
        A score of **{confidence:.2%}** means the model is highly confident that the loan status will be '{'Default' if prediction[0] == 1 else 'Paid'}'.
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("Disclaimer: This prediction is based on a machine learning model and should be used for informational purposes only. It is not financial advice.")
