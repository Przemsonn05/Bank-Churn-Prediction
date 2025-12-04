import pandas as pd
import numpy as np
import joblib
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score
from .config import RANDOM_STATE, MODELS_DIR
import os

def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def train_logistic_regression(X_train, y_train):
    print("--- Training Logistic Regression ---")
    
    pipe_lr = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model_lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
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
    
    grid_log = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_log,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_log.fit(X_train, y_train)
    print(f"Best params LogReg: {grid_log.best_params_}")
    return grid_log.best_estimator_

def train_random_forest(X_train_smote, y_train_smote):
    print("--- Training Random Forest ---")
    
    rfc_model = RandomForestClassifier(random_state=42) # Random state z inputu
    
    param_grid = {
        'n_estimators': [200, 400, 600],  
        'max_depth': [10, 15, 20, None],   
        'max_features': ['sqrt', 'log2', 0.3, 0.5], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
        'max_samples': [0.8, 0.9, None],  
        'criterion': ['gini', 'entropy'],       
        'min_impurity_decrease': [0, 0.0001, 0.001], 
        'bootstrap': [True],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rfc_model,
        param_distributions=param_grid,
        n_iter=50,  
        cv=cv,
        scoring='f1', 
        refit='f1', 
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True 
    )
    
    random_search.fit(X_train_smote, y_train_smote)
    print(f"Best params RF: {random_search.best_params_}")
    return random_search.best_estimator_

def train_lightgbm_optuna(X_train, y_train, n_trials=110):
    print("--- Training LightGBM with Optuna ---")
    
    class_weight_ratio = float(np.sum(y_train == 0)) / float(np.sum(y_train == 1))
    
    STATIC_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
    }

    def objective(trial):
        optuna_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators': trial.suggest_categorical('n_estimators', [600, 1000, 2000]),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', -1, 25),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 0.1, log=True),
            'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, np.sqrt(class_weight_ratio), class_weight_ratio])
        }

        full_params = {**STATIC_PARAMS, **optuna_params}
        model = LGBMClassifier(**full_params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=0)
                ]
            )
            
            probs = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, probs, pos_label=1)
            
            denominator = precision + recall
            f1_scores_pr = np.divide(
                2 * (precision * recall), 
                denominator, 
                out=np.zeros_like(denominator), 
                where=denominator!=0
            )
            
            f1_scores.append(np.max(f1_scores_pr))

        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best params LightGBM:", study.best_params)
    
    final_params = {**STATIC_PARAMS, **study.best_params}
    best_lgbm_model = LGBMClassifier(**final_params)
    best_lgbm_model.fit(X_train, y_train)
    
    return best_lgbm_model