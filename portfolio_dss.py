import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Loan DSS Portfolio", layout="wide")

# Custom CSS for better visuals
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #0f172a;
            padding-top: 20px;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #475569;
            margin-bottom: 40px;
        }
        .section-title {
            font-size: 28px;
            color: #1e293b;
            border-bottom: 2px solid #0ea5e9;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        .highlight {
            background-color: #ecfeff;
            padding: 10px;
            border-left: 5px solid #0ea5e9;
            margin-bottom: 20px;
        }
        .footer {
            margin-top: 60px;
            text-align: center;
            color: #64748b;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-title">📊 Loan Default Decision Support System</div>
<div class="sub-title">Built Using the DIKW & MCDA Frameworks | Lending Club Dataset (2M+ records)</div>
""", unsafe_allow_html=True)

# Project Overview
st.markdown("<div class='section-title'>🧠 Project Overview</div>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
This decision support system (DSS) helps financial institutions reduce default risk by transforming raw loan data into actionable insights using the **DIKW (Data → Information → Knowledge → Wisdom)** framework, and evaluates borrowers using **MCDA (Multi-Criteria Decision Analysis)**.
</div>
""", unsafe_allow_html=True)

# Motivation
st.markdown("<div class='section-title'>💡 Motivation</div>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
While moving to Germany, we experienced how difficult and important loan approvals are — especially for students and immigrants. That real-world challenge inspired us to build this model to support transparent, explainable credit decisions.
</div>
""", unsafe_allow_html=True)

# Methodology
st.markdown("<div class='section-title'>🛠️ Methodology</div>", unsafe_allow_html=True)

st.markdown("""
**🔹 DIKW Framework:**
- **Data:** Cleaned and filtered the Lending Club dataset with 105+ fields
- **Information:** Performed EDA on income, FICO, loan purpose, grade, interest rate, etc.
- **Knowledge:** Identified patterns behind defaults (e.g., low FICO, high DTI)
- **Wisdom:** Made actionable recommendations for lenders

**🔹 MCDA Implementation:**
- Weighted scoring of borrowers on FICO, income, DTI, employment
- Risk-based ranking: **Approve**, **Review**, or **Reject**
""")

# Tools
st.markdown("<div class='section-title'>🧰 Tools & Technologies</div>", unsafe_allow_html=True)
st.markdown("""
- Python: Pandas, Seaborn, Matplotlib
- DIKW & MCDA Frameworks
- Streamlit for portfolio visualization *(Power BI skipped due to data size)*
""")

# Team Contributions
st.markdown("<div class='section-title'>👥 Team Contributions</div>", unsafe_allow_html=True)
st.markdown("""
| Team Member | Responsibility |
|-------------|----------------|
| **Adon**     | Data cleaning and preparation |
| **Ritesh**   | Descriptive analysis & Information Layer |
| **Aarushi**  | Pattern recognition & Knowledge Layer |
| **Pratik**   | Strategic recommendations (Wisdom Layer) |
| **Flossy**   | MCDA logic & risk ranking model |
""")

# Key Outcomes
st.markdown("<div class='section-title'>📈 Key Outcomes</div>", unsafe_allow_html=True)
st.markdown("""
- Designed an explainable, transparent loan decision model
- Visualized loan behavior trends and defaults
- Created borrower risk categories for practical decisions
- Delivered a real-world simulation with >2 million records
""")

# Footer
st.markdown("""
<div class='footer'>Project by Team DSS | Final Submission July 2025</div>
""", unsafe_allow_html=True)
