from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_lightgbm(X_train, y_train):
    pipe = Pipeline([
        ('model_lgbm', LGBMClassifier(
            random_state=42,
            boosting_type='gbdt',
            n_jobs=-1,
            verbose=-1
        ))
    ])

    param_lgbm = {
        'model_lgbm__n_estimators': [200, 400, 800, 1200],
        'model_lgbm__learning_rate': [0.01, 0.05, 0.1],
        'model_lgbm__num_leaves': [63, 127, 255],
        'model_lgbm__max_depth': [-1, 10, 15, 20],
        'model_lgbm__min_child_samples': [10, 20, 50, 100],
        'model_lgbm__subsample': [0.5, 0.8, 1.0],
        'model_lgbm__colsample_bytree': [0.5, 0.8, 1.0],
        'model_lgbm__reg_alpha': [0, 0.001, 0.01, 0.1],
        'model_lgbm__reg_lambda': [0, 0.001, 0.01, 0.1]
    }

    search = RandomizedSearchCV(
        pipe,
        param_lgbm,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    return search.best_estimator_