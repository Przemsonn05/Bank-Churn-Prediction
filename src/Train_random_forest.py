from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, scoring):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    model = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring=scoring,
        refit='f1',
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_