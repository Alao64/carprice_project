import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Fit each model with its default params, score on the test set. No tuning."""
    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        report[name] = (r2, rmse)
    return report


def tune_model(model, param_grid, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """Tune one model with RandomizedSearchCV (refit on rmse), return the tuned model + test scores."""
    rs = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv,
        scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'},
        refit='rmse', random_state=42, n_jobs=-1
    )
    rs.fit(X_train, y_train)
    tuned_model = rs.best_estimator_
    y_pred = tuned_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return tuned_model, r2, rmse
