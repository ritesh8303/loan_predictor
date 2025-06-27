import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Decision Support: Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# --- Caching ---
# Cache the model and columns to avoid reloading on every interaction
@st.cache_resource
def load_artifacts():
    """Loads the pipeline and training columns."""
    pipeline = joblib.load('loan_pipeline.pkl')
    training_columns = joblib.load('training_columns.pkl')
    return pipeline, training_columns

pipeline, training_columns = load_artifacts()

# --- Helper Functions ---
def get_feature_importance(pipeline, columns):
    """Extracts feature importances from the XGBoost classifier in the pipeline."""
    # The preprocessor creates new column names for one-hot encoded features
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    # Combine numerical and new OHE feature names
    all_feature_names = np.concatenate([pipeline.named_steps['preprocessor'].transformers[0][2], ohe_feature_names])
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df

# --- Main Application ---
st.title("💡 From Data to Wisdom: A Loan Default Decision Support System")
st.markdown("""
This application uses a Machine Learning model (XGBoost) to predict the probability of a loan being fully repaid or charged off. 
It serves as a decision support tool for credit policy, aligning with the **Wisdom** stage of the DIKW framework.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header(" Simulate a New Loan Application")
st.sidebar.markdown("Enter the applicant's details below.")

# Create input fields based on the most important features
# These were identified during the analysis phase
loan_amnt = st.sidebar.slider("Loan Amount ($)", 500, 40000, 15000, 500)
term = st.sidebar.selectbox("Loan Term", [36, 60])
grade = st.sidebar.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
home_ownership = st.sidebar.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
annual_inc = st.sidebar.slider("Annual Income ($)", 10000, 300000, 75000, 1000)
purpose = st.sidebar.selectbox("Purpose of Loan", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical'])
dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 18.0, 0.1)
revol_util = st.sidebar.slider("Revolving Line Utilization (%)", 0.0, 100.0, 50.0, 0.1)
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.1)
emp_length = st.sidebar.slider("Employment Length (Years)", 0, 10, 5, 1)


# --- Prediction Button and Logic ---
if st.sidebar.button("📊 Predict Loan Repayment", use_container_width=True):
    # 1. Create a dictionary from the inputs
    # The keys must match the column names from your training data
    input_data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': np.nan,  # Model needs this column, but we can approximate or let it be imputed if pipeline handles it
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': 'Verified', # Assuming a default value
        'purpose': purpose,
        'dti': dti,
        'delinq_2yrs': 0, # Assuming default
        'inq_last_6mths': 1, # Assuming default
        'open_acc': 10, # Assuming default
        'pub_rec': 0, # Assuming default
        'revol_bal': 15000, # Assuming default
        'revol_util': revol_util,
        'total_acc': 25, # Assuming default
        'initial_list_status': 'f', # Assuming default
        'collections_12_mths_ex_med': 0, # Assuming default
        'application_type': 'Individual', # Assuming default
        'acc_now_delinq': 0, # Assuming default
        'tot_coll_amt': 0, # Assuming default
        'tot_cur_bal': 150000, # Assuming default
        'total_rev_hi_lim': 30000, # Assuming default
        'pub_rec_bankruptcies': 0 # Assuming default
    }
    
    # 2. Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])
    
    # 3. Ensure the DataFrame has the same columns in the same order as the training data
    # We fill missing columns with 0 or a suitable default. `reindex` is perfect for this.
    final_df = input_df.reindex(columns=training_columns, fill_value=0)
    
    # 4. Make prediction
    prediction = pipeline.predict(final_df)[0]
    prediction_proba = pipeline.predict_proba(final_df)[0]
    
    # --- Display Results ---
    st.subheader("🔮 Prediction Result")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("✅ **Likely to be Fully Paid**")
        else:
            st.error("❌ **High Risk: Likely to be Charged Off**")
            
    with col2:
        repayment_prob = prediction_proba[1] # Probability for class '1' (Fully Paid)
        st.metric(label="Probability of Full Repayment", value=f"{repayment_prob:.2%}")
        st.progress(repayment_prob)
        
    # --- Visual Storytelling: Feature Importance ---
    st.subheader("🧠 Why did the model make this prediction?")
    st.markdown("The chart below shows the general importance of each factor in the model's decision-making process.")
    
    feature_importance_df = get_feature_importance(pipeline, training_columns)
    
    st.bar_chart(feature_importance_df.head(10).set_index('feature'))


# --- Action Stage: MCDA ---
st.subheader("Wisdom & Action: Strategic Recommendations (MCDA)")
with st.expander("Expand to see the Multi-Criteria Decision Analysis"):
    st.markdown("""
    Based on the model's insights and Pratik's diagnostic analysis, we can compare strategic alternatives. 
    A **Weighted Scoring Model** helps us decide the best path forward.
    """)
    
    # MCDA Data
    mcda_data = {
        'Criterion': ['Risk Reduction (40%)', 'Profitability (30%)', 'Implementation Cost (20%)', 'Customer Impact (10%)'],
        'A1: Tighten Criteria': [9, 3, 8, 2],
        'A2: Implement ML Model': [8, 7, 4, 7],
        'A3: Financial Literacy Campaigns': [4, 6, 3, 9]
    }
    mcda_df = pd.DataFrame(mcda_data).set_index('Criterion')
    
    # Calculate scores
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    scores = mcda_df.values.T.dot(weights)
    
    mcda_df.loc['**Weighted Score**'] = scores
    
    st.dataframe(mcda_df.style.highlight_max(axis=1, subset=['A1: Tighten Criteria', 'A2: Implement ML Model', 'A3: Financial Literacy Campaigns'], color='lightgreen').format('{:.1f}'))
    
    st.info("""
    **Recommendation:** **Alternative 2: Implement the ML Model** is the recommended strategy.
    It provides the best balance of precise risk reduction and sustained profitability, achieving the highest weighted score of **6.8**.
    """)

