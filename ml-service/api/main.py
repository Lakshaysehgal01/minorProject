"""FastAPI prediction service for the Smart Energy Microgrid project.


Run from the `ml-service/` directory: uvicorn api.main:app --reload
"""
from contextlib import asynccontextmanager


import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODELS_DIR = "models"


models = {}




@asynccontextmanager
async def lifespan(app: FastAPI):
    models["solar"] = joblib.load(f"{MODELS_DIR}/best_solar_model.pkl")
    models["consumption"] = joblib.load(f"{MODELS_DIR}/best_consumption_model.pkl")
    yield
    models.clear()




app = FastAPI(title="Smart Energy Microgrid Prediction API", lifespan=lifespan)




class SolarRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    household_id: int = Field(..., ge=0)
    temperature: float
    solar_irradiance: float = Field(..., ge=0, le=1)
    panel_capacity_kw: float = Field(..., gt=0)




class ConsumptionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    household_id: int = Field(..., ge=0)
    temperature: float




class SolarPrediction(BaseModel):
    predicted_solar_generation_kwh: float




class ConsumptionPrediction(BaseModel):
    predicted_consumption_kwh: float




@app.get("/health")
def health():
    return {"status": "ok"}




@app.post("/predict/solar", response_model=SolarPrediction)
def predict_solar(request: SolarRequest):
    features = pd.DataFrame(
        [
            {
                "hour": request.hour,
                "day_of_week": request.day_of_week,
                "household_id": request.household_id,
                "temperature": request.temperature,
                "solar_irradiance": request.solar_irradiance,
                "panel_capacity_kw": request.panel_capacity_kw,
            }
        ]
    )
    prediction = models["solar"].predict(features)[0]
    return SolarPrediction(predicted_solar_generation_kwh=float(prediction))




@app.post("/predict/consumption", response_model=ConsumptionPrediction)
def predict_consumption(request: ConsumptionRequest):
    features = pd.DataFrame(
        [
            {
                "hour": request.hour,
                "day_of_week": request.day_of_week,
                "household_id": request.household_id,
                "temperature": request.temperature,
            }
        ]
    )
    prediction = models["consumption"].predict(features)[0]
    return ConsumptionPrediction(predicted_consumption_kwh=float(prediction))



