"""FastAPI prediction service for the Smart Energy Microgrid project.


Run from the `ml-service/` directory: uvicorn api.main:app --reload


Supports a `?model=` query param on /predict/solar and /predict/consumption:
- "xgboost" / "lightgbm": single-row request body (SolarRequest/ConsumptionRequest,
  same shape as before).
- "lstm" / "ensemble": require a DIFFERENT, sequence-shaped body - 24 hours of
  history (SolarSequenceRequest/ConsumptionSequenceRequest) - since a single
  row has no history to feed the LSTM. This is an intentional, documented
  scoping decision for this demo project, not a unified production API design.
- "prophet": not servable (it's a 3-household benchmark, not a pooled
  deployable model) - returns 400.


Any model/artifact that isn't available on this server (e.g. tensorflow not
installed, or a training script hasn't been run yet) makes only the affected
endpoints return 503 - it doesn't stop the whole service from starting.
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Literal


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator


_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT_DIR)  # for `from train import ...`
sys.path.insert(0, os.path.join(_ROOT_DIR, "models"))  # for `from train_lstm import ...`


from train import (  # noqa: E402
    CONSUMPTION_FEATURES,
    CONSUMPTION_TARGET,
    SOLAR_FEATURES,
    SOLAR_TARGET,
)
from train_lstm import LSTM_CONFIGS, WINDOW_SIZE, add_cyclical_features  # noqa: E402


try:
    from tensorflow import keras


    TENSORFLOW_AVAILABLE = True
except ImportError:
    keras = None
    TENSORFLOW_AVAILABLE = False


MODELS_DIR = "models"
SLUGS = {SOLAR_TARGET: "solar", CONSUMPTION_TARGET: "consumption"}


models = {}




def try_load(path):
    return joblib.load(path) if os.path.exists(path) else None




def try_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)




def try_load_keras(path):
    if not (TENSORFLOW_AVAILABLE and os.path.exists(path)):
        return None
    return keras.models.load_model(path)




@asynccontextmanager
async def lifespan(app: FastAPI):
    for slug in ("solar", "consumption"):
        models[f"xgboost_{slug}"] = try_load(f"{MODELS_DIR}/xgboost_{slug}_model.pkl")
        models[f"lightgbm_{slug}"] = try_load(f"{MODELS_DIR}/lightgbm_{slug}_model.pkl")


        models[f"lstm_{slug}"] = try_load_keras(f"{MODELS_DIR}/lstm_{slug}_model.keras")
        models[f"lstm_{slug}_target_scaler"] = try_load(f"{MODELS_DIR}/lstm_{slug}_target_scaler.pkl")
        models[f"lstm_{slug}_features_scaler"] = try_load(f"{MODELS_DIR}/lstm_{slug}_features_scaler.pkl")


        models[f"ridge_{slug}_meta"] = try_load(f"{MODELS_DIR}/ridge_{slug}_meta.pkl")
        models[f"ensemble_{slug}_config"] = try_load_json(f"{MODELS_DIR}/ensemble_{slug}_config.json")
        models[f"ensemble_xgboost_{slug}"] = try_load(f"{MODELS_DIR}/ensemble_xgboost_{slug}_model.pkl")
        models[f"ensemble_lightgbm_{slug}"] = try_load(f"{MODELS_DIR}/ensemble_lightgbm_{slug}_model.pkl")


    if not TENSORFLOW_AVAILABLE:
        print("[api] tensorflow not installed - lstm/ensemble endpoints will return 503.")
    yield
    models.clear()




app = FastAPI(title="Smart Energy Microgrid Prediction API", lifespan=lifespan)


ModelChoice = Literal["xgboost", "lightgbm", "lstm", "ensemble", "prophet"]




class SolarRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_year: int = Field(..., ge=1, le=365)
    month: int = Field(..., ge=1, le=12)
    household_id: int = Field(..., ge=0)
    temperature: float
    solar_irradiance: float = Field(..., ge=0, le=1)
    panel_capacity_kw: float = Field(..., gt=0)




class ConsumptionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    household_id: int = Field(..., ge=0)
    temperature: float
    consumption_profile_encoded: int = Field(..., ge=0, le=2)




class SolarSequenceRow(SolarRequest):
    # The actual observed value for that past hour - needed as the LSTM's lag
    # feature. Reasonable to require since these are 24 already-elapsed hours,
    # not future/predicted ones.
    solar_generation_kwh: float = Field(..., ge=0)




class ConsumptionSequenceRow(ConsumptionRequest):
    consumption_kwh: float = Field(..., ge=0)




class SolarSequenceRequest(BaseModel):
    rows: list[SolarSequenceRow]


    @field_validator("rows")
    @classmethod
    def check_length(cls, v):
        if len(v) != WINDOW_SIZE:
            raise ValueError(f"rows must contain exactly {WINDOW_SIZE} entries, got {len(v)}")
        return v




class ConsumptionSequenceRequest(BaseModel):
    rows: list[ConsumptionSequenceRow]


    @field_validator("rows")
    @classmethod
    def check_length(cls, v):
        if len(v) != WINDOW_SIZE:
            raise ValueError(f"rows must contain exactly {WINDOW_SIZE} entries, got {len(v)}")
        return v




class SolarPrediction(BaseModel):
    predicted_solar_generation_kwh: float




class ConsumptionPrediction(BaseModel):
    predicted_consumption_kwh: float




@app.get("/health")
def health():
    return {"status": "ok"}




def _predict_lstm(rows, target):
    slug = SLUGS[target]
    lstm_model = models.get(f"lstm_{slug}")
    target_scaler = models.get(f"lstm_{slug}_target_scaler")
    features_scaler = models.get(f"lstm_{slug}_features_scaler")
    if lstm_model is None or target_scaler is None or features_scaler is None:
        raise HTTPException(503, f"LSTM model/scalers for {target} are not loaded on this server.")


    config = LSTM_CONFIGS[target]
    df = pd.DataFrame([r.model_dump() for r in rows])
    df = add_cyclical_features(df)
    df[f"{target}_scaled"] = target_scaler.transform(df[[target]])
    scaled_scale_cols = [f"{c}_scaled" for c in config["scale_cols"]]
    df[scaled_scale_cols] = features_scaler.transform(df[config["scale_cols"]])


    feature_cols = config["cyclical_cols"] + scaled_scale_cols + [f"{target}_scaled"]
    tensor = df[feature_cols].to_numpy(dtype="float32").reshape(1, WINDOW_SIZE, len(feature_cols))
    pred_scaled = lstm_model.predict(tensor, verbose=0).reshape(-1, 1)
    return float(target_scaler.inverse_transform(pred_scaled).flatten()[0])




def _predict_ensemble(rows, target, lstm_pred):
    slug = SLUGS[target]
    ensemble_config = models.get(f"ensemble_{slug}_config")
    ridge = models.get(f"ridge_{slug}_meta")
    if ensemble_config is None or ridge is None:
        raise HTTPException(503, f"Ensemble meta-learner for {target} is not loaded on this server.")


    base_model_order = ensemble_config["base_model_order"]
    causal_features = SOLAR_FEATURES if target == SOLAR_TARGET else CONSUMPTION_FEATURES
    last_row_df = pd.DataFrame([rows[-1].model_dump()])[causal_features]


    preds = {"lstm": lstm_pred}
    if "xgboost" in base_model_order:
        preds["xgboost"] = float(models[f"ensemble_xgboost_{slug}"].predict(last_row_df)[0])
    if "lightgbm" in base_model_order:
        preds["lightgbm"] = float(models[f"ensemble_lightgbm_{slug}"].predict(last_row_df)[0])


    ordered = [[preds[name] for name in base_model_order]]
    return float(ridge.predict(ordered)[0])




def _predict_tabular(model_choice, target, request_row):
    slug = SLUGS[target]
    model = models.get(f"{model_choice}_{slug}")
    if model is None:
        raise HTTPException(503, f"{model_choice} model for {target} is not loaded on this server.")
    features = SOLAR_FEATURES if target == SOLAR_TARGET else CONSUMPTION_FEATURES
    frame = pd.DataFrame([request_row.model_dump()])[features]
    return float(model.predict(frame)[0])




@app.post("/predict/solar", response_model=SolarPrediction)
def predict_solar(payload: dict, model: ModelChoice = Query("ensemble")):
    if model == "prophet":
        raise HTTPException(
            400,
            "Prophet is a per-household benchmark, not available for live "
            "prediction - see results/model_comparison.py.",
        )


    if model in ("xgboost", "lightgbm"):
        prediction = _predict_tabular(model, SOLAR_TARGET, SolarRequest(**payload))
        return SolarPrediction(predicted_solar_generation_kwh=prediction)


    if not TENSORFLOW_AVAILABLE:
        raise HTTPException(503, "LSTM/ensemble models are unavailable (tensorflow not installed).")


    seq = SolarSequenceRequest(**payload)
    lstm_pred = _predict_lstm(seq.rows, SOLAR_TARGET)
    if model == "lstm":
        return SolarPrediction(predicted_solar_generation_kwh=lstm_pred)


    ensemble_pred = _predict_ensemble(seq.rows, SOLAR_TARGET, lstm_pred)
    return SolarPrediction(predicted_solar_generation_kwh=ensemble_pred)




@app.post("/predict/consumption", response_model=ConsumptionPrediction)
def predict_consumption(payload: dict, model: ModelChoice = Query("ensemble")):
    if model == "prophet":
        raise HTTPException(
            400,
            "Prophet is a per-household benchmark, not available for live "
            "prediction - see results/model_comparison.py.",
        )


    if model in ("xgboost", "lightgbm"):
        prediction = _predict_tabular(model, CONSUMPTION_TARGET, ConsumptionRequest(**payload))
        return ConsumptionPrediction(predicted_consumption_kwh=prediction)


    if not TENSORFLOW_AVAILABLE:
        raise HTTPException(503, "LSTM/ensemble models are unavailable (tensorflow not installed).")


    seq = ConsumptionSequenceRequest(**payload)
    lstm_pred = _predict_lstm(seq.rows, CONSUMPTION_TARGET)
    if model == "lstm":
        return ConsumptionPrediction(predicted_consumption_kwh=lstm_pred)


    ensemble_pred = _predict_ensemble(seq.rows, CONSUMPTION_TARGET, lstm_pred)
    return ConsumptionPrediction(predicted_consumption_kwh=ensemble_pred)



