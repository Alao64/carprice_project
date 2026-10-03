# 🚗 Car Price Predictor

A machine learning web app that estimates used car prices from listing details, built on a tuned ensemble of gradient-boosted and tree-based regressors, with per-prediction explainability powered by SHAP.

**[Live Demo](https://predictedcarprice.streamlit.app/)**

---

## Overview

This project takes a raw used-car listings dataset through a full ML pipeline involving cleaning, feature engineering, model selection, hyperparameter tuning, and ensembling and serves the final model through an interactive Streamlit app. Rather than relying on a single model, the top 3 performers out of 6 candidate algorithms are tuned individually and combined into a weighted voting ensemble, balancing accuracy with robustness.

Every prediction is paired with a SHAP-based breakdown showing the top 5 features that pushed that specific estimate above or below the model's baseline — so the app explains its reasoning, not just its output.

## Features

- **6-model evaluation → top-3 selection → hyperparameter tuning → weighted VotingRegressor**
  Random Forest, Linear Regression, Ridge, Lasso, Elastic Net, XGBoost, AdaBoost, and CatBoost are evaluated; the strongest 3 (by RMSE) are tuned with `RandomizedSearchCV` and combined into an inverse-RMSE-weighted ensemble.
- **Per-prediction explainability**
  SHAP `TreeExplainer` on the best individual tuned model shows the top 5 features driving each specific price estimate, with real input values (not just feature names).
- **Price range, not a bare number**
  Estimates are shown with a ± RMSE range and a comparison against the average price for similar listings (same brand/model type) in the training data.
- **Interactive Streamlit UI**
  Tabbed form (Basic Info / Specs / History / Features), styled result card, glowing animated predict button, dark theme.

## Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.972 |
| RMSE | ~$1,406 |
| Ensemble | CatBoost, XGBoost, Random Forest (weighted by inverse tuned RMSE) |

## Runtime

Full pipeline (data ingestion → transformation → 6-model evaluation → top-3 tuning → voting ensemble): **~8 minutes** on a standard machine (CPU-only, `n_jobs=-1`).
Inference (single prediction + SHAP explanation) is near-instant (<1 second).

## Tech Stack

- **Modeling:** scikit-learn, XGBoost, CatBoost, SHAP
- **App:** Streamlit
- **Data handling:** pandas, NumPy
- **Persistence:** joblib

## Project Structure

```
carprice_project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Loads, cleans, and splits the raw dataset
│   │   ├── data_transformation.py  # Preprocessing pipeline (scaling + one-hot encoding)
│   │   └── model_trainer.py        # 6-model eval → top-3 tuning → voting ensemble
│   ├── pipeline/
│   │   └── predict_pipeline.py     # Inference + SHAP explanation for the Streamlit app
│   ├── exception.py
│   ├── logger.py
│   └── utils.py                    # save_object, evaluate_models, tune_model
├── artifacts/                      # Trained model, preprocessor, and processed data (tracked in repo)
├── notebook/                       # Original dataset and exploratory notebook
├── app.py                          # Streamlit application
├── requirements.txt
└── setup.py
```

## Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Alao64/carprice_project.git
   cd carprice_project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Train the model** (runs data ingestion → transformation → model training, saves artifacts):
```bash
python -m src.components.data_ingestion
```

**Run the app:**
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## How It Works

1. **Data Ingestion** — raw listings are cleaned (typo correction, type coercion, feature engineering such as car age and brand/type extraction from model names).
2. **Data Transformation** — numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded.
3. **Model Training** — all 6 candidate models are evaluated with default parameters; the top 3 by RMSE are tuned via `RandomizedSearchCV`; a weighted `VotingRegressor` combines them based on inverse tuned RMSE.
4. **Inference & Explainability** — the saved `VotingRegressor` produces the price estimate; a separately saved single best-performing tree model powers SHAP explanations for that same prediction.

## Author

**Abdulrahman** —  Data scientist
[GitHub](https://github.com/Alao64) · 
