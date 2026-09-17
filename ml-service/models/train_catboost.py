

"""Train CatBoost regressors for solar generation and consumption.


Run from the `ml-service/` directory: python models/train_catboost.py


Reuses the same time-based split, feature lists, and evaluation helpers as
train.py (the Phase 1 baseline trainer) so this is a like-for-like addition
to the model comparison, not a separate pipeline.
"""
import os
import sys


import joblib
import pandas as pd
from catboost import CatBoostRegressor


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import (  # noqa: E402
    CONSUMPTION_FEATURES,
    CONSUMPTION_TARGET,
    DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SOLAR_FEATURES,
    SOLAR_TARGET,
    TARGET_SLUGS,
    evaluate,
    time_based_split,
    write_metrics,
)


MODEL_NAME = "CatBoostRegressor"




def train_and_evaluate(df, target, features, label):
    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]


    # n_estimators=200 matches XGBoost/LightGBM's setting for a like-for-like
    # comparison (CatBoost's sklearn wrapper accepts n_estimators as an alias
    # for its native "iterations"). verbose=False suppresses per-iteration
    # training logs, CatBoost's equivalent of LightGBM's verbosity=-1.
    model = CatBoostRegressor(n_estimators=200, random_state=RANDOM_STATE, verbose=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)
    write_metrics(target, MODEL_NAME, metrics)


    # Persisted so the API can serve ?model=catboost directly.
    joblib.dump(model, f"{MODELS_DIR}/catboost_{TARGET_SLUGS[target]}_model.pkl")


    print(f"=== CatBoost for {label} ({target}) ===")
    print(
        f"RMSE={metrics['RMSE']:.4f} MSE={metrics['MSE']:.4f} "
        f"MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
    )
    return model, metrics




def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    train_and_evaluate(df, SOLAR_TARGET, SOLAR_FEATURES, "solar generation")
    train_and_evaluate(df, CONSUMPTION_TARGET, CONSUMPTION_FEATURES, "consumption")




if __name__ == "__main__":
    main()



