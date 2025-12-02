from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def train_logistic_regression(X_train, y_train):
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=101)),
        ('model_lr', LogisticRegression(max_iter=1000, random_state=101))
    ])

    param_log = [
        {
            "model_lr__solver": ["liblinear"],
            "model_lr__penalty": ["l1", "l2"],
            "model_lr__C": [0.001, 0.01, 0.1, 1, 10]
        },
        {
            "model_lr__solver": ["saga"],
            "model_lr__penalty": ["l1", "l2"],
            "model_lr__C": [0.001, 0.01, 0.1, 1, 10]
        },
        {
            "model_lr__solver": ["saga"],
            "model_lr__penalty": ["elasticnet"],
            "model_lr__l1_ratio": [0, 0.3, 0.5, 0.7, 1],
            "model_lr__C": [0.001, 0.01, 0.1, 1, 10]
        }
    ]

    grid = GridSearchCV(pipe_lr, param_log, scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_