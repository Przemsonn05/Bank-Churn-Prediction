import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
from .config import IMAGES_DIR
import os

def save_plot(fig, filename):
    """Zapisuje wykres do folderu images."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {path}")

def evaluate_model_performance(model, X_test, y_test, model_name):
    """
    Standardowa ewaluacja: Classification Report, ROC-AUC score, Confusion Matrix.
    """
    print(f"\n=== Evaluating {model_name} ===")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    save_plot(plt.gcf(), f'Confusion_Matrix_{model_name.replace(" ", "_")}.png')
    
    return y_proba, auc_score

def analyze_coefficients_logreg(model, feature_names):
    """Analiza współczynników dla Regresji Logistycznej."""
    if 'model_lr' in model.named_steps:
        coefs = model.named_steps['model_lr'].coef_[0]
    else:
        coefs = model.coef_[0]

    coeffs_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefs
    })
    coeffs_df = coeffs_df.sort_values(by='Weight', ascending=False)
    
    print("\nTop Logistic Regression Coefficients:")
    print(coeffs_df.head(10))
    print(coeffs_df.tail(10))
    return coeffs_df

def plot_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    features_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
    plt.figure(figsize=(12, 6))
    features_series.plot(kind='bar', color='teal')
    plt.title(f'Feature Importance – {model_name}')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    
    save_plot(plt.gcf(), f'Feature_importance_{model_name.replace(" ", "_")}.png')

def plot_shap_summary(model, X_test, model_name):
    """Generuje wykres SHAP summary."""
    print(f"Generating SHAP summary for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    save_plot(plt.gcf(), f'Feature_value_{model_name.replace(" ", "_")}.png')

def compare_models_roc(models_dict, X_test, y_test):
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC-AUC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    save_plot(plt.gcf(), 'ROC_AUC_comparison.png')