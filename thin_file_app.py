import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Page Configuration
st.set_page_config(page_title="CreditLens AI - Alternative Credit Scorer", page_icon="💳", layout="centered")

st.title("💳 CreditLens: Multi-Source Alternative Credit Scorer")
st.write("Research Prototype: Evaluating thin-file borrowers using multi-source alternative telemetry.")

st.markdown("---")
st.subheader("📊 Applicant Behavioral & Telemetry Profile")

col1, col2 = st.columns(2)

with col1:
    account_age = st.slider(
        "Telecom Account Age (Months)", 1, 36, 12,
        help="How long your mobile telecom account has been active."
    )
    recharge_avg = st.number_input(
        "Monthly Recharge Average (₹)", min_value=100, max_value=1500, value=350,
        help="Average amount spent monthly on mobile recharges."
    )
    recharge_delay = st.slider(
        "Recharge Delay Days", 0, 15, 2,
        help="Average number of days delayed before renewing prepaid plans."
    )

with col2:
    data_stability = st.slider(
        "Data Usage Stability (0.0 to 1.0)", 0.1, 1.0, 0.85,
        help="Consistency of daily mobile internet usage patterns."
    )
    utility_rate = st.slider(
        "Utility On-Time Payment Rate (0.0 to 1.0)", 0.3, 1.0, 0.90,
        help="How regularly household utility bills (electricity, water) are paid on time."
    )
    upi_count = st.number_input(
        "Monthly UPI Transaction Count", min_value=2, max_value=50, value=20,
        help="Frequency of digital cash-flow / UPI transactions per month."
    )

# Model Training
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Account_Age_Months': np.random.randint(1, 36, n_samples),
        'Monthly_Recharge_Avg': np.random.uniform(100, 1500, n_samples),
        'Recharge_Delay_Days': np.random.randint(0, 15, n_samples),
        'Data_Usage_Stability': np.random.uniform(0.1, 1.0, n_samples),
        'Utility_OnTime_Payment_Rate': np.random.uniform(0.3, 1.0, n_samples),
        'UPI_Monthly_Txn_Count': np.random.randint(2, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df['Credit_Eligible'] = np.where(
        (df['Recharge_Delay_Days'] < 5) & 
        (df['Account_Age_Months'] > 6) & 
        (df['Utility_OnTime_Payment_Rate'] > 0.6) & 
        (df['UPI_Monthly_Txn_Count'] > 8), 1, 0
    )
    
    X = df[['Account_Age_Months', 'Monthly_Recharge_Avg', 'Recharge_Delay_Days', 
            'Data_Usage_Stability', 'Utility_OnTime_Payment_Rate', 'UPI_Monthly_Txn_Count']]
    y = df['Credit_Eligible']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = get_trained_model()

st.markdown("<br>", unsafe_allow_html=True)

# Assessment Calculation
if st.button("🔍 Run Comprehensive Credit Assessment", use_container_width=True):
    input_data = np.array([[account_age, recharge_avg, recharge_delay, data_stability, utility_rate, upi_count]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.markdown("---")
    st.subheader("🎯 Assessment Results & Insights")
    
    if prediction[0] == 1:
        confidence = prediction_proba[0][1] * 100
        st.success(f"Status: **LOAN APPROVED (Low Risk)** — Confidence Score: {confidence:.2f}%")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Estimated Credit Limit", "₹45,000", delta="+₹10,000 bonus limit")
        with col_res2:
            st.metric("Suggested Interest Rate", "11.5% p.a.", delta="-2% lower than standard")
            
        st.info("📌 **Why you qualified:** Your low recharge delay and strong utility payment discipline show high reliability.")
    else:
        confidence = prediction_proba[0][0] * 100
        st.error(f"Status: **ADDITIONAL VERIFICATION REQUIRED** — Confidence Score: {confidence:.2f}%")
        
        st.markdown("#### 🛠️ How to Improve Your Score:")
        st.markdown("""
        * **Reduce Recharge Delays:** Try to top up your mobile plan before it expires.
        * **Maintain Utility Timeliness:** Ensure electricity and water bills are cleared promptly.
        * **Active Digital Footprint:** Regular UPI transactions boost your cash-flow reliability profile.
        """)

# --- ENHANCED RESEARCH FEEDBACK MODULE ---
st.markdown("---")
st.subheader("📝 Model Evaluation & Research Feedback")
st.write("Help us record user perception and model reliability metrics for our research analysis.")

with st.form("feedback_form"):
    user_role = st.selectbox("1. Select Your Profile:", ["Student", "Fintech Professional", "Lending Expert", "General User"])
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fairness_rating = st.slider("2. Rate Fairness & Transparency (1 to 5)", 1, 5, 4)
        usefulness_score = st.slider("3. Model Usefulness for Thin-File Borrowers (1 to 5)", 1, 5, 4)
    with col_f2:
        trust_score = st.slider("4. Rate Trustworthiness of Telecom/UPI Data (1 to 5)", 1, 5, 4)
        data_pref = st.selectbox("5. Perception on Alternative Telemetry vs Traditional Bureau Score:", [
            "Highly Reliable Alternative", 
            "Good Supplementary Data", 
            "Needs Traditional Credit History"
        ])
        
    comments = st.text_area("6. Any additional feedback or improvement suggestions?")
    
    submitted = st.form_submit_button("Submit Feedback to Research Log", use_container_width=True)
    
    if submitted:
        feedback_file = "research_responses.csv"
        new_data = pd.DataFrame([{
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Role": user_role,
            "Fairness_Rating": fairness_rating,
            "Trust_Score": trust_score,
            "Usefulness_Score": usefulness_score,
            "Alternative_Data_Perception": data_pref,
            "Comments": comments
        }])
        
        if os.path.exists(feedback_file):
            new_data.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            new_data.to_csv(feedback_file, mode='w', header=True, index=False)
            
        st.success("Thank you! Your response has been securely logged for our research report.")

# Live Analytics Dashboard
if os.path.exists("research_responses.csv"):
    df_results = pd.read_csv("research_responses.csv")
    if not df_results.empty:
        st.markdown("---")
        st.subheader("📊 Live Research Analytics Summary")
        st.write(f"Total Peer Responses Logged: **{len(df_results)}**")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Avg Fairness", round(df_results["Fairness_Rating"].mean(), 2))
        with col_m2:
            st.metric("Avg Trust", round(df_results["Trust_Score"].mean(), 2))
        with col_m3:
            st.metric("Avg Usefulness", round(df_results["Usefulness_Score"].mean(), 2))
