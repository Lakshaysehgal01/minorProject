"""Stacking ensemble: XGBoost + LSTM (+ every candidate booster - currently
LightGBM and CatBoost - that individually beats XGBoost), blended by a Ridge
meta-learner, for both solar_generation_kwh and consumption_kwh.


Run from the `ml-service/` directory: python models/train_ensemble.py


Requires `models/train_lightgbm.py` and `models/train_catboost.py` to have
already been run (their metrics JSON files are read to decide whether each
one beats XGBoost and should be added as a base model) and
`models/train_lstm.py` to have already been run (this script loads and
reuses that pretrained LSTM rather than retraining one).


NOTE: like train_lstm.py, this requires `tensorflow` and could only be
static-checked (not executed) on the machine this was written on.


Prophet is deliberately NOT a base model here (see models/train_prophet.py's
docstring) - it's a per-household benchmark evaluated on only 3 of the 25
households, and mixing a per-household model into a pooled ensemble would
mean the ensemble could only be evaluated on those same 3 households,
undermining the "pooled, all-25-household" comparison this ensemble
represents.


Methodology: base models are trained on the 70% TRAIN block of a 3-way
train/validation/test day split (train.time_based_split_3way), their
predictions on the 10% VALIDATION block train a Ridge meta-learner, and the
full stack is evaluated on the 20% TEST block - a base model's own training
data never leaks into what trains or evaluates the meta-learner.


Row alignment: the LSTM has no predictions for the first 24 hours of each
household's period (window requirement), so all base model predictions are
merged on (household_id, timestamp) via an inner join - this naturally trims
the tabular models' predictions down to the LSTM's available rows, without
relying on fragile positional slicing.
"""
import json
import os
import sys


import joblib
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


try:
    from tensorflow import keras
except ImportError:
    keras = None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)  # for `from train import ...`
sys.path.insert(0, _THIS_DIR)  # for `from train_lstm import ...` (sibling module)


from train import (  # noqa: E402
    CONSUMPTION_FEATURES,
    CONSUMPTION_TARGET,
    DATA_PATH,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SOLAR_FEATURES,
    SOLAR_TARGET,
    evaluate,
    time_based_split_3way,
    write_metrics,
)
from train_lstm import LSTM_CONFIGS, add_cyclical_features, build_dataset  # noqa: E402


FEATURES_BY_TARGET = {SOLAR_TARGET: SOLAR_FEATURES, CONSUMPTION_TARGET: CONSUMPTION_FEATURES}


# Candidate tabular boosters considered as extra ensemble base models, beyond
# the always-included XGBoost + LSTM. Each is included independently if it
# beats XGBoost's Phase-1/Phase-2 R2 for that target - not capped at one, so
# both could end up included, or neither. Adding a third booster later only
# needs a new entry here, not new branching logic below.
CANDIDATE_BOOSTERS = {
    "lightgbm": {
        "metric_slug": "lgbmregressor",
        "build": lambda: LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=-1),
    },
    "catboost": {
        "metric_slug": "catboostregressor",
        "build": lambda: CatBoostRegressor(n_estimators=200, random_state=RANDOM_STATE, verbose=False),
    },
}




def booster_beats_xgb(metric_slug, target):
    xgb_path = os.path.join(METRICS_DIR, f"{target}__xgbregressor.json")
    candidate_path = os.path.join(METRICS_DIR, f"{target}__{metric_slug}.json")
    if not (os.path.exists(xgb_path) and os.path.exists(candidate_path)):
        return False
    with open(xgb_path) as f:
        xgb_r2 = json.load(f)["r2"]
    with open(candidate_path) as f:
        candidate_r2 = json.load(f)["r2"]
    return candidate_r2 > xgb_r2




def predict_tabular_df(model, df, features):
    return pd.DataFrame(
        {
            "household_id": df["household_id"].to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
            "pred": model.predict(df[features]),
        }
    )




def predict_lstm_df(model, target_scaler, splits, split_name):
    X = splits[split_name]["X"]
    keys = splits[split_name]["keys"]
    if len(X) == 0:
        return pd.DataFrame(columns=["household_id", "timestamp", "pred"])
    preds_scaled = model.predict(X, verbose=0).reshape(-1, 1)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    return pd.DataFrame(
        {
            "household_id": [k[0] for k in keys],
            "timestamp": [k[1] for k in keys],
            "pred": preds,
        }
    )




def build_aligned_predictions(base_preds, actual_df):
    """Inner-join every base model's predictions plus the actual target value
    on (household_id, timestamp), so every row has a prediction from every
    base model and a known label. Used identically for validation (to train
    the meta-learner) and test (to evaluate it).
    """
    merged = None
    for name, pred_df in base_preds.items():
        renamed = pred_df.rename(columns={"pred": f"pred_{name}"})
        merged = (
            renamed
            if merged is None
            else pd.merge(merged, renamed, on=["household_id", "timestamp"], how="inner")
        )
    merged = pd.merge(merged, actual_df, on=["household_id", "timestamp"], how="inner")
    return merged




def run_for_target(target, df):
    if keras is None:
        raise RuntimeError("tensorflow is not installed - cannot load/run the LSTM base model.")
    features = FEATURES_BY_TARGET[target]
    config = LSTM_CONFIGS[target]
    slug = config["slug"]


    train_df, val_df, test_df = time_based_split_3way(df)


    # --- Base model 1: XGBoost, freshly fit on this 3-way split's train block
    # (Phase 1's XGBoost was fit on the 2-way 80/20 split's train block, which
    # overlaps this split's validation block - refitting avoids that leak). ---
    xgb_model = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0)
    xgb_model.fit(train_df[features], train_df[target])


    base_val_preds = {"xgboost": predict_tabular_df(xgb_model, val_df, features)}
    base_test_preds = {"xgboost": predict_tabular_df(xgb_model, test_df, features)}


    # Persisted separately from train.py's xgboost_{slug}_model.pkl (that one is
    # fit on the 2-way 292-day split): the ensemble's Ridge meta-learner was
    # calibrated on THIS model's validation predictions, so serving must reuse
    # this exact model, not the standalone one, to avoid train/serve skew.
    joblib.dump(xgb_model, f"{MODELS_DIR}/ensemble_xgboost_{slug}_model.pkl")


    # --- Base models 2+ (conditional): each candidate booster (LightGBM,
    # CatBoost, ...), included independently if it beat XGBoost in its own
    # Phase 2 comparison (models/train_lightgbm.py / train_catboost.py must
    # have already run). ---
    for name, spec in CANDIDATE_BOOSTERS.items():
        included = booster_beats_xgb(spec["metric_slug"], target)
        print(f"[{target}] {name} included as base model: {included}")
        if included:
            booster_model = spec["build"]()
            booster_model.fit(train_df[features], train_df[target])
            base_val_preds[name] = predict_tabular_df(booster_model, val_df, features)
            base_test_preds[name] = predict_tabular_df(booster_model, test_df, features)
            joblib.dump(booster_model, f"{MODELS_DIR}/ensemble_{name}_{slug}_model.pkl")


    # --- Base model 3: pretrained pooled LSTM (loaded, not retrained here).
    # Rebuilding the same 3-way split + feature pipeline here is deterministic
    # (train_lstm.build_dataset fits its scalers on the train block, which is
    # bit-for-bit identical to what train_lstm.py itself used), so this
    # reproduces the exact scaling the loaded model expects without needing to
    # separately persist/load scaler files across scripts. ---
    lstm_model = keras.models.load_model(f"{MODELS_DIR}/lstm_{slug}_model.keras")
    labeled_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    labeled_df = add_cyclical_features(labeled_df)
    splits, target_scaler, _features_scaler, _feature_cols = build_dataset(labeled_df, target)


    base_val_preds["lstm"] = predict_lstm_df(lstm_model, target_scaler, splits, "val")
    base_test_preds["lstm"] = predict_lstm_df(lstm_model, target_scaler, splits, "test")


    # --- Align on (household_id, timestamp), fit Ridge meta-learner on val,
    # evaluate the full stack on test. ---
    val_actual = val_df[["household_id", "timestamp", target]].rename(columns={target: "actual"})
    test_actual = test_df[["household_id", "timestamp", target]].rename(columns={target: "actual"})


    val_aligned = build_aligned_predictions(base_val_preds, val_actual)
    test_aligned = build_aligned_predictions(base_test_preds, test_actual)


    pred_cols = [f"pred_{name}" for name in base_val_preds]
    meta = Ridge()
    meta.fit(val_aligned[pred_cols], val_aligned["actual"])


    test_preds = meta.predict(test_aligned[pred_cols])
    metrics = evaluate(test_aligned["actual"].to_numpy(), test_preds)
    write_metrics(target, "Ensemble", metrics)


    joblib.dump(meta, f"{MODELS_DIR}/ridge_{slug}_meta.pkl")


    # Records which base models (and in what order) the meta-learner expects -
    # api/main.py reads this at startup so it knows whether to load an
    # ensemble_lightgbm_{slug}_model.pkl and what column order to feed Ridge.
    with open(f"{MODELS_DIR}/ensemble_{slug}_config.json", "w") as f:
        json.dump({"base_model_order": list(base_val_preds)}, f, indent=2)


    print(
        f"Ensemble ({target}) base models={list(base_val_preds)}: "
        f"RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
    )




def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    run_for_target(SOLAR_TARGET, df)
    run_for_target(CONSUMPTION_TARGET, df)




if __name__ == "__main__":
    main()



