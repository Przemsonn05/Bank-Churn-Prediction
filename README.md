# Bank Churn Prediction 🏦📉

End-to-end Data Science project aiming to predict customer churn in the banking sector using machine learning.

The primary business objective was to build a model that identifies customers at risk of leaving, while maintaining a strategic balance between **Recall** (catching churners) and **Precision** (avoiding false alarms) to optimize retention campaign budgets.

---

## 📌 Key Results & Business Recommendations

After experimenting with multiple algorithms, **LightGBM** was selected as the production model ("Champion Model").

| Metric | Score (Test Set) | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.89** | Excellent ability to distinguish between loyal and churning customers. |
| **Recall** | **66%** | The model detects nearly 2/3 of all customers who are actually leaving. |
| **Precision** | **64%** | When the model flags a risk, it is correct 64% of the time (minimizing "spam" and costs). |

### What drives churn? (Insights)
Model analysis (SHAP & Feature Importance) revealed critical risk factors:
1.  **Number of Products:** Customers with **3 or 4 products** are in the critical risk group. However, having exactly 2 products significantly increases loyalty.
2.  **Age:** The **45-60 age demographic** shows a much higher tendency to resign.
3.  **Activity:** Inactive members (`IsActiveMember = 0`) are significantly more likely to churn.

**Strategic Recommendation:**
Focus retention efforts on **inactive clients aged 45+ holding >2 products**. Due to the model's high precision, the bank can safely offer this segment higher-value incentives (e.g., cash bonuses) with minimal risk of wasting budget on customers who would have stayed anyway.

---

## 📊 Visualizations

### 1. Model Decision Factors (SHAP Values)
The plot below illustrates how specific features impact the probability of churn.

![SHAP Summary Plot](images/Feature_value_model3.png)

### 2. Model Effectiveness (Confusion Matrix)
The model effectively minimizes false positives, accurately identifying high-risk customers while avoiding unnecessary interventions for loyal clients, thereby making retention campaigns more cost-efficient and maximizing the bank’s overall return on investment.

![Confusion Matrix](images/Confusion_Matrix_model3.png)

---

🔄 End-to-End ML Pipeline

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
      │  Model Explainability│
      └─────────────────────┘
```

---

## ⚙️ Methodology & Process

The project was executed in the following stages:

1.  **Exploratory Data Analysis (EDA):**
    * Identified a non-linear relationship between product count and churn.
    * Detected a strong correlation between age and customer decisions.
2.  **Data Preprocessing:**
    * Scaling numerical features (`StandardScaler`).
    * One-Hot Encoding for categorical variables (Geography, Gender).
    * Dropping irrelevant identifiers (`CustomerId`, `Surname`).
3.  **Handling Class Imbalance:**
    * The dataset was imbalanced (80% Retained / 20% Churned).
    * Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training set to prevent data leakage.
4.  **Modeling & Evaluation:**
    * **Baseline:** Logistic Regression (High Recall, but very low Precision - too "aggressive").
    * **Challenger:** Random Forest (Good performance, handled non-linearity well).
    * **Final:** **LightGBM** + Hyperparameter Tuning (Optuna). Achieved the best F1-Score balance.

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