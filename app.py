import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIGURATION ---
# This must be the very first streamlit command
st.set_page_config(page_title="Churn Analysis App", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .card {
        padding: 30px;
        border-radius: 15px;
        background: rgba(102, 126, 234, 0.15);
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.35);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(118, 75, 162, 0.35);
        text-align: center;
        margin-bottom: 20px;
        transition: 0.3s ease;
    }

    /* Hover effect */
    .card:hover {
        background: rgba(118, 75, 162, 0.25);
        box-shadow: 0 10px 40px rgba(118, 75, 162, 0.45);
        transform: translateY(-4px);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
    }
    
    /* Title styling */
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def yes_no_to_int(value):
    """Converts 'Yes'/'No' to 1/0"""
    return 1 if value == "Yes" else 0

# --- MODEL LOADING ---
class DummyModel:
    """
    A placeholder model to prevent the app from crashing 
    if the actual model file is missing during testing.
    """
    def predict_proba(self, data):
        # Returns a random probability for demonstration
        # [probability_class_0, probability_class_1]
        val = np.random.uniform(0, 1)
        return [[1-val, val]]

@st.cache_resource
def load_model():
    """Loads the model or returns a dummy if not found."""
    model_path = "models/churn_lightgbm_model.joblib"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return DummyModel()
    else:
        st.warning(f"⚠️ Model file not found at '{model_path}'. Using Dummy Model for UI demonstration.")
        return DummyModel()

model = load_model()

# --- NAVIGATION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# --- HOME PAGE ---
def home_page():
    st.title("🏠 Customer Churn Analysis App")
    st.write("### Select an option below to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>📊 Calculate Client Churn</h3>
            <p>Predict the probability of customer churn</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Calculator", key="btn1", use_container_width=True):
            navigate_to('calculate')
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>🧠 How the Model Works</h3>
            <p>Learn about the prediction model</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Explanation", key="btn2", use_container_width=True):
            navigate_to('explain')
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3>💼 Business Support</h3>
            <p>Business insights and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Recommendations", key="btn3", use_container_width=True):
            navigate_to('recommendations')

    st.divider()

    st.subheader("About This Project")
    st.markdown("Feel free to explore the features above! 😊")

# --- CALCULATOR PAGE ---
def calculate_page():
    st.title("Calculate Client Churn")
    
    if st.button("← Back to Home"):
        navigate_to('home')
    
    st.divider()
    
    st.subheader("Enter Customer Information:")
    
    # Input Layout
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Is_active_member = st.selectbox("Is Active Member", ["No", "Yes"])
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    
    with col2:
        Number_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        Tenure = st.number_input("Tenure (months)", min_value=0, max_value=10, value=1)
        Has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
        Credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600, step=1)

    # Full width sliders for monetary values
    st.write("") # Spacer
    col3, col4 = st.columns(2)
    with col3:
        Estimated_salary = st.slider("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=50000.0, step=500.0)
    with col4:
        Balance = st.slider("Account Balance ($)", min_value=0.0, max_value=250000.0, value=0.0, step=500.0)
    
    st.write("---")
    
    if st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True):
        # Prepare input data matching model features
        input_data = pd.DataFrame({
            "CreditScore": [Credit_score],
            "Geography_Germany": [1 if Geography == "Germany" else 0],
            "Geography_Spain": [1 if Geography == "Spain" else 0],
            "Geography_France": [1 if Geography == "France" else 0],
            "Gender_Male": [1 if Gender == "Male" else 0],
            "Gender_Female": [1 if Gender == "Female" else 0],
            "Age": [Age],
            "Tenure": [Tenure],
            "Balance": [Balance],
            "NumOfProducts": [Number_of_products],
            "HasCrCard": [yes_no_to_int(Has_cr_card)],
            "IsActiveMember": [yes_no_to_int(Is_active_member)],
            "EstimatedSalary": [Estimated_salary],
        })
        
        try:
            # Predict
            prediction_proba = model.predict_proba(input_data)[0]
            churn_probability = prediction_proba[1] * 100
            
            # Display Results
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Churn Probability", f"{churn_probability:.1f}%")
            with col_res2:
                if churn_probability > 70:
                    st.error("HIGH RISK - Immediate action required")
                elif churn_probability > 40:
                    st.warning("MEDIUM RISK - Monitor closely")
                else:
                    st.success("LOW RISK - Customer is stable")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            with st.expander("See Input Data Debug"):
                st.write(input_data)

# --- EXPLAIN PAGE ---
def explain_page():
    st.title("How the Model Works")
    
    if st.button("← Back to Home"):
        navigate_to('home')
    
    st.divider()

    st.markdown("### Methodology & Process")
    st.write("The model was built using a rigorous data science pipeline to ensure accuracy and reliability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Data Preprocessing**
        * **Scaling:** Numerical features were scaled using `RobustScaler` to normalize range.
        * **Encoding:** Categorical variables (Geography, Gender) were converted using One-Hot Encoding.
        * **Cleaning:** Irrelevant identifiers (IDs, Surnames) were removed.
        """)
        
    with col2:
        st.markdown("""
        **2. Handling Imbalance (SMOTE)**
        * The original data was imbalanced (80% Retained / 20% Churned).
        * We applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training set to prevent the model from being biased toward the majority class.
        """)

    st.divider()

    st.markdown("### The Champion Model: LightGBM")
    st.write("After testing Logistic Regression (Baseline) and Random Forest (Challenger), **LightGBM** was selected as the production model due to its superior balance of precision and speed.")

    # Metrics Display
    st.markdown("#### Performance Metrics (Test Set)")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("ROC-AUC Score", "0.88")
        st.caption("Excellent ability to distinguish between loyal and churning customers.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Recall", "62%")
        st.caption("Detects nearly 2/3 of all customers who are actually leaving.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Precision", "67.5%")
        st.caption("Minimizes 'spam'—when the model predicts churn, it is correct 67.5% of the time.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    
    st.markdown("### Model Interpretability")
    st.info("To understand *why* the model makes a decision, we analyzed Feature Importance and SHAP values.")
    
    st.write("")
    
    st.markdown("""
    * **Feature Importance:** The model identified that **Product Count**, **Age**, and **Member Activity** are the strongest predictors.
    * **Confusion Matrix:** The model effectively minimizes false positives, allowing for cost-efficient retention campaigns.
    """)

# --- RECOMMENDATIONS PAGE ---
def recommendations_page():
    st.title("Business Support & Insights")
    
    if st.button("← Back to Home"):
        navigate_to('home')
    
    st.divider()
    
    st.subheader("Executive Summary")
    st.write("""
    Based on the model analysis, we have identified distinct customer cohorts with high churn risk. 
    Addressing these specific groups yields the highest Return on Investment (ROI) for retention budgets.
    """)
    
    st.markdown("### What Drives Churn?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Product Holdings")
        st.write("Customers with **3 or 4 products** are in the critical risk group.")
        st.markdown("**Insight:** Having exactly **2 products** significantly increases loyalty.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Demographics")
        st.write("The **45-60 age demographic** shows a much higher tendency to resign.")
        st.markdown("**Insight:** Older age pushes the decision probability towards Churn.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Member Activity")
        st.write("**Inactive members** (`IsActiveMember = 0`) are significantly more likely to churn.")
        st.markdown("**Insight:** Engagement is a leading indicator of retention.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.divider()

    st.markdown("### Strategic Recommendations")
    
    # Strategy Box
    st.markdown("""
    <div style='background-color: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #667eea;'>
        <h4>Primary Strategy: Precision Targeting</h4>
        <p>
            Focus retention efforts specifically on <b>Inactive clients aged 45+ holding >2 products</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    #### Recommended Actions:
    1.  **High-Value Incentives:** Due to the model's high precision (67.5%), we can safely offer this segment higher-value incentives (e.g., cash bonuses, rate reductions) with minimal risk of wasting budget on customers who would have stayed anyway.
    2.  **Product Consolidation:** For customers with 3-4 products, review if the bundle is too complex or expensive. Attempt to restructure them into the "sweet spot" of 2 products.
    3.  **Re-engagement Campaigns:** Launch specific email/app campaigns targeting the 45-60 age group to flip their status from "Inactive" to "Active."
    """)

# === ROUTING ===
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'calculate':
    calculate_page()
elif st.session_state.page == 'explain':
    explain_page()
elif st.session_state.page == 'recommendations':
    recommendations_page()