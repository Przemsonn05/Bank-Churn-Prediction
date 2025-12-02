from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
import shap
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()

def shap_summary(model, X_test):
    explainer = shap.TreeExplainer(model.named_steps['model_lgbm'])
    shap_vals = explainer.shap_values(X_test)
    shap.summary_plot(shap_vals, X_test)
    return shap_vals