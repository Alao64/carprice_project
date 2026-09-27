from src.logger import logging
import os
import sys
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object, evaluate_models, tune_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
                "Elastic Net": ElasticNet(alpha=0.01, max_iter=10000),
                "XGBRegressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                "CatBoost": CatBoostRegressor(verbose=False, random_state=42),
            }

            params = {
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
                    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['sqrt', 'log2', None],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"],
                },
                "CatBoost": {
                    "iterations": [200, 500, 800],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7],
                },
            }

            # Step 1: evaluate all 6 with default params
            initial_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            sorted_initial = sorted(initial_report.items(), key=lambda x: x[1][1])  # ascending rmse
            logging.info("Initial evaluation (all 6 models):")
            for name, (r2, rmse) in sorted_initial:
                logging.info(f"{name}: R2 = {r2:.4f} | RMSE = {rmse:,.2f}")

            top3_names = [name for name, _ in sorted_initial[:3]]

            # Step 2: tune the top 3
            tuned = {}
            for name in top3_names:
                tuned_model, r2, rmse = tune_model(
                    models[name], params.get(name, {}), X_train, y_train, X_test, y_test
                )
                tuned[name] = {"model": tuned_model, "r2": r2, "rmse": rmse}
                logging.info(f"Tuned {name}: R2 = {r2:.4f} | RMSE = {rmse:,.2f}")

            # Step 3: weights from inverse RMSE (lower rmse -> higher weight)
            inv_rmse = {name: 1.0 / tuned[name]["rmse"] for name in top3_names}
            total = sum(inv_rmse.values())
            weights = [inv_rmse[name] / total for name in top3_names]

            # Step 4: VotingRegressor on the tuned models
            estimators = [(name, tuned[name]["model"]) for name in top3_names]
            voting_model = VotingRegressor(estimators=estimators, weights=weights)
            voting_model.fit(X_train, y_train)

            y_test_pred = voting_model.predict(X_test)
            voting_r2 = r2_score(y_test, y_test_pred)
            voting_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Identify the single best of the top 3 by RMSE, save it separately for SHAP
            best_individual_name = min(top3_names, key=lambda name: tuned[name]["rmse"])
            best_individual_model = tuned[best_individual_name]["model"]

            save_object(
                file_path=os.path.join("artifacts", "best_individual_model.pkl"),
                obj=best_individual_model,
                        )
            logging.info(f"Best individual model for SHAP: {best_individual_name} | RMSE = {tuned[best_individual_name]['rmse']:,.2f}")

            if voting_r2 < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"VotingRegressor ({', '.join(top3_names)}) | "
                f"R2 = {voting_r2:.4f} | RMSE = {voting_rmse:,.2f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=voting_model,
            )

            tuned_scores = [(name, tuned[name]["r2"], tuned[name]["rmse"]) for name in top3_names]
            print("DEBUG tuned_scores:", tuned_scores)
            print("DEBUG type:", type(tuned_scores), [type(t) for t in tuned_scores])

            
            return voting_r2, voting_rmse, tuned_scores

        except Exception as e:
            raise CustomException(e, sys)