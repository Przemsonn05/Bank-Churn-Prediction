import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Churn Analysis App", layout="wide")

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
        background: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def yes_no_to_int(value):
    """Converts 'Yes'/'No' to 1/0"""
    return 1 if value == "Yes" else 0

class DummyModel:
    """
    A placeholder model to prevent the app from crashing 
    if the actual model file is missing during testing.
    """
    def predict_proba(self, data):
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

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

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

    st.markdown("""
        <div style="text-align: center;">
            <h3>About This Project</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center;">
    This project is an end-to-end <b>Bank Churn Prediction</b> analysis aimed at understanding customer behavior and identifying those at risk of leaving a bank.<br><br>

    <b>Bank churn</b> refers to the phenomenon where customers stop using a bank’s services and move to a competitor. Predicting churn is crucial for financial institutions because retaining an existing customer is typically much more cost-effective than acquiring a new one. Understanding churn patterns also helps banks offer personalized services and improve customer satisfaction.<br><br>

    <b>Key Highlights of This Project:</b><br>
    • <b>Objective:</b> Build a predictive model that accurately identifies high-risk customers, balancing metrics like <b>Recall</b> and <b>Precision</b> to make retention campaigns cost-effective.<br>
    • <b>Data Analysis & Feature Engineering:</b> Extensive Exploratory Data Analysis (EDA), handling outliers, correlations, and feature engineering like <i>Balance-to-Salary Ratio</i> and <i>CreditScore-to-Age Ratio</i>.<br>
    • <b>Modeling & Machine Learning:</b> Tested Logistic Regression, Random Forest, and <b>LightGBM</b> with hyperparameter tuning. Evaluated using F1-Score, ROC-AUC, and confusion matrices.<br>
    • <b>Handling Class Imbalance:</b> Applied <b>SMOTE</b> strictly on training data to improve detection of churners.<br>
    • <b>Visualization & Interpretability:</b> Used <b>Seaborn, Matplotlib, and SHAP</b> for feature importance, customer behavior trends, and model decisions.<br>
    • <b>Technology Stack:</b> Python 3.12, <b>Pandas, NumPy, Scikit-Learn, LightGBM, Imbalanced-learn, Joblib</b>, and <b>Streamlit</b> for interactive dashboards.<br><br>

    This project demonstrates the technical steps in predictive modeling while providing actionable business insights. Customers aged 45+ with multiple products who are inactive are more likely to churn, enabling targeted retention strategies.<br><br>

    Feel free to explore the interactive visualizations, model predictions, and feature insights above! 😊
    </div>
    """, unsafe_allow_html=True)

def calculate_page():
    st.title("Calculate Client Churn")
    
    if st.button("← Back to Home"):
        navigate_to('home')
    
    st.divider()
    
    st.subheader("Enter Customer Information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=18)
        Gender = st.selectbox("Gender", ["Male", "Female"], None)
        Is_active_member = st.selectbox("Is Active Member", ["No", "Yes"], None)
    
    with col2:
        Tenure = st.number_input("Tenure (months)", min_value=0, max_value=10, value=1)
        Number_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], None)
        Has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"], None)

    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], None)
    Credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=0, step=1)
    Estimated_salary = st.slider("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=0.0, step=500.0)
    Balance = st.slider("Account Balance ($)", min_value=0.0, max_value=250000.0, value=0.0, step=500.0)
    
    st.divider()
    
    if st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True):
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
            prediction_proba = model.predict_proba(input_data)[0]
            churn_probability = prediction_proba[1] * 100
            
            #st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Churn Probability", f"{churn_probability:.1f}%")
            with col_res2:
                if churn_probability > 65:
                    st.error("HIGH RISK - Action required")
                elif churn_probability > 50:
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
        * **Scaling:** Numerical features were scaled using RobustScaler to normalize range.
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
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("ROC-AUC Score", "0.89")
        st.caption("Excellent ability to distinguish between loyal and churning customers.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col2:
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Recall", "66%")
        st.caption("Detects nearly 2/3 of all customers who are actually leaving.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col3:
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Precision", "64%")
        st.caption("Minimizes 'spam'—when the model predicts churn, it is correct 64% of the time.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    
    st.markdown("### Model Interpretability")
    st.info("To understand *why* the model makes a decision, we analyzed Feature Importance and SHAP values.")
    
    st.write("")
    
    st.markdown("""
    * **Feature Importance:** Analysis using SHAP values revealed that **Product Count**, **Age**, and **Member Activity** are the most influential features driving churn predictions.  
    - Customers with **3–4 products** show a higher churn risk, while those with exactly **2 products** demonstrate strong loyalty.  
    - **Age** highlights a critical risk group between **45–60 years**, likely reflecting life-stage financial decisions.  
    - **Inactive members** (IsActiveMember = 0) are significantly more likely to churn, emphasizing the need for engagement campaigns.

    * **Confusion Matrix & Model Performance:** The model achieves a balanced trade-off between **Recall** and **Precision**, effectively identifying high-risk customers while minimizing false positives.  
    - This ensures **cost-efficient retention campaigns** by targeting customers who are genuinely at risk, avoiding unnecessary incentives to loyal clients.  
    - The resulting predictions can inform **strategic segmentation**, prioritizing interventions for the most vulnerable demographic.
    """)

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
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Product Holdings")
        st.write("Customers with **3 or 4 products** are in the critical risk group.")
        st.markdown("**Insight:** Having exactly **2 products** significantly increases loyalty.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Demographics")
        st.write("The **45-60 age demographic** shows a much higher tendency to resign.")
        st.markdown("**Insight:** Older age pushes the decision probability towards Churn.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        #st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Member Activity")
        st.write("**Inactive members** (IsActiveMember = 0) are significantly more likely to churn.")
        st.markdown("**Insight:** Engagement is a leading indicator of retention.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### Strategic Recommendations")
    
    st.markdown("""
    <div style='background-color: rgba(102, 126, 234, 0.1); padding: 25px; border-radius: 10px; border-left: 5px solid #667eea;'>
        <h4>Precision Targeting: The "Silent Attrition" Cohort</h4>
        <p style='font-size: 1.1em;'>
            <b>Target Segment:</b> Inactive clients | Age 45-60 | Holding 3+ Products
        </p>
        <p>
            Our LightGBM model identifies the intersection of these three features as the <b>highest probability churn cluster</b>. 
            These are often financially mature clients who are likely dissatisfied with product complexity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    #### Recommended Actions:
    
    1.  **The "2-Product Sweet Spot" Strategy:** * *Insight:* Data shows churn drops significantly for customers with exactly 2 products.
        * *Action:* For clients with 3-4 products, offer a **Product Consolidation Bundle**. Simplify their portfolio (e.g., merge separate savings accounts) to reduce fees and management effort.
    
    2.  **High-Touch Re-engagement for Age 45+:**
        * *Insight:* This demographic is less responsive to generic app notifications but values relationship banking.
        * *Action:* Trigger a personal call from a relationship manager focusing on "Financial Health Review" rather than a sales pitch.
    
    3.  **Leverage High Precision:**
        * *Insight:* The model has ~64% Precision, meaning false positives are low.
        * *Action:* Authorize a higher retention budget for this specific group, knowing the ROI will be positive because the risk is real.
    """)

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'calculate':
    calculate_page()
elif st.session_state.page == 'explain':
    explain_page()
elif st.session_state.page == 'recommendations':
    recommendations_page()