# 🏦 Bank Customer Churn Prediction

This project implements an **end-to-end Machine Learning solution** designed to predict customer attrition in the banking sector. Going beyond simple classification, the primary objective was to build a profit-driven model that identifies high-risk customers while optimizing the trade-off between **Recall** (capturing churners) and **Precision** (minimizing retention costs).

Powered by a hyper-tuned **LightGBM** model and deployed via a **Streamlit** dashboard, this solution provides actionable insights to support proactive retention strategies and budget optimization.

---

## 📌 Key Results & Business Recommendations

After experimenting with multiple algorithms, **LightGBM** was selected as the production model ("Champion Model").

| Metric | Score (Test Set) | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.89** | Excellent ability to distinguish between loyal and churning customers. |
| **Recall** | **66%** | The model detects nearly 2/3 of all customers who are actually leaving. |
| **Precision** | **64%** | When the model flags a risk, it is correct 64% of the time (minimizing "spam" and costs). |

## 🔍 Key Drivers of Churn
Comprehensive model interpretability analysis (utilizing SHAP values and Feature Importance) revealed the following primary determinants of customer attrition:

1. **Product Portfolio Composition (Primary Driver)**

- High Risk: Customers holding 3-4 products exhibit drastically elevated churn probability
  
- Optimal Engagement: Holding exactly 2 products correlates strongly with retention, suggesting optimal service utilization
  
- Insight: Product proliferation may indicate customer confusion or dissatisfaction rather than engagement

2. **Demographic Risk Factors**

- Age Cohort: The 45-60 age group demonstrates the highest propensity to churn
  
- Geography: German customers show significantly higher churn rates compared to French and Spanish customers

- Action: Develop age-specific value propositions for mid-to-late career professionals

3. **Member Engagement Status**

- Critical Indicator: Inactive members (IsActiveMember = 0) are primary churn candidates
  
- Protective Factor: Active engagement serves as a significant retention buffer
  
- Recommendation: Implement proactive re-engagement campaigns for dormant accounts

4. **Financial Indicators**

- Balance per Product: Higher balance concentrated in fewer products indicates stronger engagement
  
- Credit Score: Surprisingly low predictive power, suggesting churn is more behavioral than credit-risk driven
  
- Salary-to-Balance Ratio: Customers who maintain high balances relative to income show stronger loyalty

## Strategic Recommendation

- Implement a High-Precision Retention Strategy: Resource allocation should prioritize inactive clients aged 45+ holding >2 products.

- Rationale: Leveraging the model's high precision (64%), the bank can deploy high-value intervention tactics (e.g., financial incentives, premium support) with confidence.

- ROI Impact: This targeted approach mitigates the risk of resource misallocation (wasting budget on customers likely to stay) while maximizing the retention of high-value, at-risk accounts.

---

## 📊 Visualizations

### 1. Model Decision Factors (SHAP Values)
The plot below illustrates how specific features impact the probability of churn.

![SHAP Summary Plot](images/Feature_value_model3.png)

Key Insights from SHAP Analysis:

- NumOfProducts and Age are the dominant predictors of customer churn
  
- IsActiveMember status critically influences predictions—inactivity strongly pushes toward churn
  
- Geography (particularly Germany) significantly elevates risk
  
- Balance and Balance_per_Product show nuanced effects:

  - Higher absolute balance correlates with increased churn risk

  - Higher balance per product (concentrated engagement) reduces risk

- CreditScore exhibits surprisingly low predictive power, indicating churn is primarily behavioral

### 2. Model Performance (Confusion Matrix)
   
The confusion matrix demonstrates the model's effectiveness in minimizing false positives while maintaining strong recall:

![Confusion Matrix](images/Confusion_Matrix_model3.png)

Performance Breakdown:

- ✅ True Negatives (1,413): Correctly identified loyal customers (84% specificity)
  
- ✅ True Positives (276): Correctly identified churning customers (66% recall)
  
- ⚠️ False Positives (164): Loyal customers flagged as at-risk (acceptable trade-off for retention)
  
- ❌ False Negatives (147): Missed churners (34% slip-through rate)

---

## 🔄 End-to-End ML Pipeline

The project covers the complete ML development cycle:

```powershell
Data → EDA → Preprocessing → Feature Engineering → 
Train/Test Split → SMOTE → Modeling → Evaluation → Explainability
```

```powershell
        ┌────────────┐
        │   Raw Data │
        └─────┬──────┘
              ▼
     ┌───────────────────┐
     │ Exploratory Data  │
     │     Analysis      │
     └─────────┬─────────┘
               ▼
  ┌──────────────────────────┐
  │  Preprocessing & Feature │
  │      Engineering         │
  └────────────┬─────────────┘
               ▼
   ┌────────────────────────┐
   │ Train / Test Split     │
   │  + SMOTE on training   │
   └───────────┬────────────┘
               ▼
     ┌──────────────────────┐
     │   Model Training     │
     │  (Baseline → LGBM)   │
     └──────────┬───────────┘
                ▼
      ┌─────────────────────┐
      │   Evaluation +      │
      │ Model Explainability│
      └─────────────────────┘
```

---

## ⚙️ Methodology & Technical Approach

1. **Deep-Dive Exploratory Data Analysis (EDA)**

- Multivariate Analysis: Uncovered non-linear relationships, identifying the "2-product sweet spot" for customer retention
  
- Distribution Diagnostics: Diagnosed significant right-skewness in financial features (Balance, EstimatedSalary)
  
- Risk Cohort Identification: Detected high-risk segments (inactive members aged 45-60)
  
- Outlier Strategy: Retained financial outliers as they represent high-net-worth individuals critical to business

2. **Advanced Feature Engineering and Preprocessing Techniques**

- BalanceSalaryRatio: Identifies customers who treat the bank as their primary savings institution
  
- BalancePerProduct: Distinguishes engaged clients from "phantom" users (many products, low balance)
  
- TenurePerAge: Captures relationship maturity relative to customer lifecycle stage

- RobustScaler: Mitigates outlier impact without information loss (better than StandardScaler for financial data)
  
- Native Categorical Support: Leveraged LightGBM's built-in categorical handling (more efficient than One-Hot Encoding)

3. **Imbalanced Data Handling**

- SMOTE (Synthetic Minority Over-sampling Technique): Addressed severe class imbalance (80:20 ratio)
  
- Leakage Prevention: Implemented within imblearn.pipeline to ensure synthetic samples generated only during training phase
  
- Validation Integrity: Preserved original distribution in validation/test sets for unbiased evaluation

4. **Model Selection & Optimization**

- Bayesian Optimization (Optuna): 200+ hyperparameter combinations explored
  
- Objective Function: Maximize F1-Score (balances precision and recall)
  
- Cross-Validation: 5-Fold Stratified CV to ensure robust generalization
  
- Threshold Tuning: Post-processing optimization via Precision-Recall curve analysis

5. **Model Explainability**

- SHAP (SHapley Additive exPlanations): Provides both global feature importance and local instance-level explanations
  
- Feature Importance: LightGBM's gain-based importance ranking

---

## 🛠️ Tech Stack

* **Python 3.12.11**
* **Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM, Imbalanced-learn (SMOTE), Joblib, Optuna
* **Visualization:** Matplotlib, Seaborn, SHAP

---

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone <repository url>
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook "Bank Churn Predictions.ipynb"
    ```
---
