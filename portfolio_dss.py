import streamlit as st

st.set_page_config(page_title="Loan Default DSS Portfolio", layout="wide")

# Header
st.title("📊 Loan Default Decision Support System")
st.subheader("Built Using the DIKW & MCDA Frameworks")

# Overview
st.markdown("""
### 🧠 Project Overview
This project aims to help financial institutions reduce loan default risk by transforming raw historical loan data into actionable decisions using the **DIKW (Data-Information-Knowledge-Wisdom)** framework and **MCDA (Multi-Criteria Decision Analysis)**.

We used Lending Club data (2007–2018) with over 2 million records to simulate a real-world loan approval process.
""")

# Motivation
st.markdown("""
### 💡 Motivation
Our inspiration came from our own financial journey — applying for loans while moving to Germany. We realized how crucial transparent and data-driven lending decisions are, especially for international students and immigrants.
""")

# Methodology
st.markdown("""
### 🧪 Methodology

#### 🔹 DIKW Framework:
- **Data:** Cleaned and preprocessed large Lending Club dataset
- **Information:** Descriptive analysis on income, loan purpose, employment, FICO score, and more
- **Knowledge:** Discovered correlations like higher default risk for small business loans
- **Wisdom:** Provided actionable recommendations for lenders

#### 🔹 MCDA:
- Used weighted scoring on multiple features (FICO, DTI, employment length, etc.)
- Created borrower risk levels: Approve / Review / Reject
""")

# Tools
st.markdown("""
### 🛠️ Tools & Technologies
- **Python** (Pandas, Seaborn, Matplotlib)
- **DIKW & MCDA Frameworks**
- *(Skipped Power BI due to dataset size)*
""")

# Team Roles
st.markdown("""
### 👥 Team Contributions
| Team Member | Role |
|-------------|------|
| **Adon** | Data cleaning and preparation |
| **Ritesh** | Descriptive analysis & Information layer |
| **Aarushi** | Pattern recognition & Knowledge layer |
| **Pratik** | Strategic recommendations (Wisdom layer) |
| **Flossy** | MCDA model design and borrower scoring |
""")

# Outcomes
st.markdown("""
### ✅ Key Outcomes
- Transparent, interpretable decision support system
- Borrower risk-ranking using MCDA
- Strategic recommendations for safer lending
- Real-world simulation using 2M+ loan records
""")

# Footer
st.markdown("---")
st.markdown("Project by Team DSS | July 2025")
