"""Train a pooled LSTM (all 25 households, one model per target) on sliding
24-hour windows.


Run from the `ml-service/` directory: python models/train_lstm.py


NOTE: this script requires `tensorflow` and could not be smoke-tested on the
machine this was written on (TensorFlow's Windows wheel needs Windows Long
Path support, which isn't enabled here) - only static/syntax checks were
possible. Validate end-to-end on a machine with tensorflow actually installed
before relying on the results.


Design notes:
- Pooled, not per-household: a single model per target, with `household_id`
  included as a raw per-timestep feature channel (sufficient to distinguish
  25 households; an embedding input would only matter at much higher
  cardinality, and the requested architecture is a single-input Sequential
  model, not a two-branch functional model).
- hour/day_of_week/day_of_year get sin/cos cyclical encoding for this model
  specifically (unlike the tabular tree models, which keep raw ordinals) -
  an LSTM/dense layer has no reason to know hour=23 and hour=0 are adjacent
  unless told via cyclical encoding.
- Every numeric input (not just the target, per the literal ask) is
  MinMaxScaled, fit on the train split only, to keep gradient updates stable
  across differently-scaled inputs (panel_capacity_kw ~2-12 vs a 0-1 target).
- Sequences are built per household (so a window never crosses a household
  boundary) and per the 3-way train/val/test day split from
  `train.time_based_split_3way` (so a window never straddles a split
  boundary, and the LSTM's split boundaries are bit-for-bit identical to the
  stacking ensemble's - required for train_ensemble.py's row alignment).
"""
import os
import sys


import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


try:
    from tensorflow import keras
except ImportError:
    # Kept optional so callers that only need LSTM_CONFIGS/add_cyclical_features/
    # build_dataset (e.g. api/main.py, when serving without tensorflow installed)
    # can still import this module. Anything that actually builds/trains/loads a
    # model raises a clear error below instead of failing at import time.
    keras = None


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import (  # noqa: E402
    CONSUMPTION_TARGET,
    DATA_PATH,
    MODELS_DIR,
    SOLAR_TARGET,
    evaluate,
    time_based_split_3way,
    write_metrics,
)


WINDOW_SIZE = 24
EPOCHS = 50


# Per-target LSTM feature configuration. `scale_cols` are MinMax-scaled
# (fit on train only); cyclical sin/cos columns are already bounded in
# [-1, 1] by construction and need no scaler. The target's own scaled value
# is appended as the last feature column (the "lag" channel - the window's
# historical y values); see build_dataset().
LSTM_CONFIGS = {
    SOLAR_TARGET: {
        "cyclical_cols": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"],
        "scale_cols": ["temperature", "solar_irradiance", "panel_capacity_kw", "household_id"],
        "slug": "solar",
    },
    CONSUMPTION_TARGET: {
        "cyclical_cols": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        "scale_cols": ["temperature", "household_id", "consumption_profile_encoded"],
        "slug": "consumption",
    },
}




def add_cyclical_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df




def build_dataset(labeled_df, target):
    """labeled_df must already have a "split" column (train/val/test) and the
    cyclical feature columns. Returns per-split (X, y) arrays, the fitted
    target/features scalers, and the ordered feature-column list.
    """
    config = LSTM_CONFIGS[target]
    scale_cols = config["scale_cols"]
    cyclical_cols = config["cyclical_cols"]


    labeled_df = labeled_df.copy()
    train_mask = labeled_df["split"] == "train"


    target_scaler = MinMaxScaler()
    target_scaler.fit(labeled_df.loc[train_mask, [target]])
    labeled_df[f"{target}_scaled"] = target_scaler.transform(labeled_df[[target]])


    features_scaler = MinMaxScaler()
    features_scaler.fit(labeled_df.loc[train_mask, scale_cols])
    scaled_scale_cols = [f"{c}_scaled" for c in scale_cols]
    labeled_df[scaled_scale_cols] = features_scaler.transform(labeled_df[scale_cols])


    feature_cols = cyclical_cols + scaled_scale_cols + [f"{target}_scaled"]


    splits = {name: {"X": [], "y": [], "keys": []} for name in ("train", "val", "test")}


    for household_id, hh_df in labeled_df.groupby("household_id"):
        hh_df = hh_df.sort_values("timestamp").reset_index(drop=True)
        feature_matrix = hh_df[feature_cols].to_numpy(dtype="float32")
        target_vector = hh_df[f"{target}_scaled"].to_numpy(dtype="float32")
        split_labels = hh_df["split"].to_numpy()
        timestamps = hh_df["timestamp"].to_numpy()


        for i in range(WINDOW_SIZE, len(hh_df)):
            # Window is rows i-24..i-1 (past 24 hours), label is row i.
            window_splits = split_labels[i - WINDOW_SIZE : i + 1]
            if not (window_splits == window_splits[0]).all():
                continue  # straddles a train/val/test boundary - drop
            split_name = window_splits[0]
            splits[split_name]["X"].append(feature_matrix[i - WINDOW_SIZE : i])
            splits[split_name]["y"].append(target_vector[i])
            splits[split_name]["keys"].append((household_id, timestamps[i]))


    for split_name in splits:
        splits[split_name]["X"] = np.array(splits[split_name]["X"], dtype="float32")
        splits[split_name]["y"] = np.array(splits[split_name]["y"], dtype="float32")


    return splits, target_scaler, features_scaler, feature_cols




def build_model(n_features):
    if keras is None:
        raise RuntimeError("tensorflow is not installed - cannot build/train the LSTM model.")
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model




def train_and_evaluate(labeled_df, target, epochs=EPOCHS):
    config = LSTM_CONFIGS[target]
    splits, target_scaler, features_scaler, feature_cols = build_dataset(labeled_df, target)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]


    model = build_model(n_features=len(feature_cols))
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=2,
    )


    preds_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


    metrics = evaluate(y_test_actual, preds)
    write_metrics(target, "LSTM", metrics)


    slug = config["slug"]
    model.save(f"{MODELS_DIR}/lstm_{slug}_model.keras")
    joblib.dump(target_scaler, f"{MODELS_DIR}/lstm_{slug}_target_scaler.pkl")
    joblib.dump(features_scaler, f"{MODELS_DIR}/lstm_{slug}_features_scaler.pkl")


    print(
        f"LSTM ({target}): RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
        f"R2={metrics['R2']:.4f}"
    )
    return model, metrics




def main(epochs=EPOCHS):
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    train_df, val_df, test_df = time_based_split_3way(df)
    labeled_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    labeled_df = add_cyclical_features(labeled_df)


    train_and_evaluate(labeled_df, SOLAR_TARGET, epochs=epochs)
    train_and_evaluate(labeled_df, CONSUMPTION_TARGET, epochs=epochs)




if __name__ == "__main__":
    main()



