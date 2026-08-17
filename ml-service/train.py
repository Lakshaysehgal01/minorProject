"""Train and compare multiple regression models for solar generation and consumption.


Run from the `ml-service/` directory: python train.py


Uses a TIME-based train/test split (first 80% of days -> train, last 20% -> test)
since this is time series data - a random shuffle would leak future information
into the training set.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


DATA_PATH = "data/synthetic_energy_data.csv"
MODELS_DIR = "models"
TEST_FRACTION = 0.2
RANDOM_STATE = 42


SOLAR_TARGET = "solar_generation_kwh"
SOLAR_FEATURES = [
    "hour",
    "day_of_week",
    "household_id",
    "temperature",
    "solar_irradiance",
    "panel_capacity_kw",
]


CONSUMPTION_TARGET = "consumption_kwh"
CONSUMPTION_FEATURES = ["hour", "day_of_week", "household_id", "temperature"]




def time_based_split(df, test_fraction=TEST_FRACTION):
    """Split by calendar day so all households' rows for a given day stay together."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_days = pd.to_datetime(df["timestamp"]).dt.date.unique()
    unique_days.sort()
    n_train_days = int(len(unique_days) * (1 - test_fraction))
    train_days = set(unique_days[:n_train_days])


    day_series = pd.to_datetime(df["timestamp"]).dt.date
    train_mask = day_series.isin(train_days)
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)




def build_models():
    return {
        "LinearRegression": LinearRegression(),
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("svr", SVR())]
        ),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0
        ),
    }




def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MSE": mse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }




def train_and_compare(df, target, features, label):
    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]


    results = []
    fitted_models = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["Model"] = name
        results.append(metrics)
        fitted_models[name] = model


    comparison = pd.DataFrame(results)[["Model", "RMSE", "MSE", "MAE", "R2"]]
    comparison = comparison.sort_values("R2", ascending=False).reset_index(drop=True)


    print(f"\n=== Model comparison for {label} ({target}) ===")
    print(comparison.to_string(index=False))


    comparison.to_csv(f"{MODELS_DIR}/{target}_comparison.csv", index=False)


    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"Best model for {label}: {best_name} (R2 = {comparison.iloc[0]['R2']:.4f})")


    return best_name, best_model




def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


    best_solar_name, best_solar_model = train_and_compare(
        df, SOLAR_TARGET, SOLAR_FEATURES, "solar generation"
    )
    joblib.dump(best_solar_model, f"{MODELS_DIR}/best_solar_model.pkl")
    if best_solar_name == "SVR":
        joblib.dump(best_solar_model.named_steps["scaler"], f"{MODELS_DIR}/solar_scaler.pkl")


    best_consumption_name, best_consumption_model = train_and_compare(
        df, CONSUMPTION_TARGET, CONSUMPTION_FEATURES, "consumption"
    )
    joblib.dump(best_consumption_model, f"{MODELS_DIR}/best_consumption_model.pkl")
    if best_consumption_name == "SVR":
        joblib.dump(
            best_consumption_model.named_steps["scaler"],
            f"{MODELS_DIR}/consumption_scaler.pkl",
        )


    print(f"\nSaved best solar model ({best_solar_name}) to {MODELS_DIR}/best_solar_model.pkl")
    print(
        f"Saved best consumption model ({best_consumption_name}) "
        f"to {MODELS_DIR}/best_consumption_model.pkl"
    )




if __name__ == "__main__":
    main()



