# Machine Learning Service (`ml-service/`)

FastAPI microservice and training pipeline for hourly **solar generation** and **household consumption** forecasting in a 25-household smart microgrid, using synthetic telemetry conditioned on real Delhi historical weather data.

---

## 1. Architecture & Overview

The ML layer is responsible for training, comparing, and serving multiple regression models. Hourly forecasts determine which households will have an energy surplus and which will experience a deficit, directly driving the trading market in the backend.

```
                      +-----------------------------------+
                      |       Raw Data Sources            |
                      |  - Open-Meteo Delhi Weather (2024)|
                      |  - Household Profiles & Capacities|
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |      data/generate_data.py        |
                      |  Produces data/energy_data.csv    |
                      +-----------------+-----------------+
                                        |
            +---------------------------+---------------------------+
            |                           |                           |
            v                           v                           v
     +--------------+           +---------------+           +---------------+
     | Tabular Tree |           | Deep Learning |           | Meta-Learner  |
     |   Models     |           |     LSTM      |           |   Ensemble    |
     |  (train.py,  |           | (train_lstm)  |           | (train_ensemble|
     |   LightGBM,  |           |  24h sequence |           |  Ridge on top |
     |   CatBoost)  |           |  cyclical enc |           |  of base preds|
     +-------+------+           +-------+-------+           +-------+-------+
             |                          |                           |
             +--------------------------+---------------------------+
                                        | *.pkl / *.keras artifacts
                                        v
                      +-----------------------------------+
                      |      FastAPI Service (api/main.py)|
                      |  Listens on http://127.0.0.1:8000 |
                      |  Endpoints:                       |
                      |   - GET  /health                  |
                      |   - POST /predict/solar           |
                      |   - POST /predict/consumption     |
                      +-----------------------------------+
```

---

## 2. Models & Forecasting Approaches

| Model | Kind | Input Features | Notes |
|---|---|---|---|
| **Linear Regression** | Baseline | Tabular features | Fast, interpretable reference baseline |
| **Random Forest** | Tree Ensemble | Tabular features | Non-linear tree baseline |
| **XGBoost** | Gradient Boosting | Tabular features | Primary gradient boosting baseline |
| **LightGBM** | Gradient Boosting | Tabular features | Fast leaf-wise tree splitting; default production model |
| **CatBoost** | Gradient Boosting | Tabular features | Categorical-aware boosting benchmark |
| **LSTM** | Deep Learning (RNN) | 24h sequence + cyclical sin/cos encodings | Captures diurnal inertia & sequential temporal dependencies |
| **Stacking Ensemble** | Meta-Learner | Base predictions: `pred_xgboost`, `pred_lightgbm`, `pred_lstm` | Ridge regression meta-learner combining tree & deep learning predictions |
| **Prophet** | Additive Time-Series | Datetime + target | Per-household additive benchmark (`results/model_comparison.py`) |

### Feature Engineering
- **Solar target:** `hour`, `day_of_week`, `day_of_year`, `month`, `temperature`, `cloud_cover`, `humidity`, `wind_speed`, `panel_capacity_kw`.
- **Consumption target:** `hour`, `day_of_week`, `household_id`, `temperature`, `humidity`, `is_holiday`, `consumption_profile_encoded`.
- **Cyclical Temporal Features (LSTM):** Sine and cosine transforms for `hour` ($24\text{h}$ period), `day_of_week` ($7\text{d}$ period), and `day_of_year` ($365\text{d}$ period).

---

## 3. Directory Layout

```
ml-service/
├── api/
│   └── main.py                 # FastAPI application, lifespan model loader, endpoints
├── data/
│   ├── generate_data.py        # Generates synthetic microgrid dataset
│   ├── fetch_weather.py        # Open-Meteo weather API client
│   └── weather_cache/          # Cached historical Delhi weather (delhi_weather_2024.csv)
├── models/
│   ├── train_lightgbm.py       # Trains LightGBM models for solar & consumption
│   ├── train_catboost.py       # Trains CatBoost models
│   ├── train_lstm.py           # Trains pooled LSTM models using TensorFlow/Keras
│   └── train_ensemble.py       # Calibrates Ridge stacking ensemble across base models
├── results/
│   ├── model_comparison.py     # Multi-model evaluation script
│   └── *_model_comparison.csv  # Benchmark tables comparing RMSE, MAE, R²
├── train.py                    # Phase 1 baseline training script
└── requirements.txt            # Python dependencies
```

---

## 4. Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
cd ml-service
python -m venv venv

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on TensorFlow:** `tensorflow` is required for LSTM and Ensemble inference. If running without GPU, standard CPU wheels will automatically optimize instruction sets (AVX2/FMA).

---

## 5. Training Pipeline (Artifact Generation)

Generate data and train the models in this sequence (artifacts are saved to `ml-service/models/` and are gitignored):

```bash
# 1. Generate microgrid dataset (downloads Delhi weather once):
python data/generate_data.py

# 2. Train baseline models (Linear Regression, Random Forest, XGBoost):
python train.py

# 3. Train Phase 2 gradient boosters:
python models/train_lightgbm.py
python models/train_catboost.py

# 4. Train LSTM deep learning models:
python models/train_lstm.py

# 5. Calibrate the stacking ensemble meta-learner:
python models/train_ensemble.py

# 6. Generate comparative metrics for the frontend:
python results/model_comparison.py
```

---

## 6. Running the API

Start the service using `uvicorn`:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

#### `GET /health`
Returns service status and indicates whether models are loaded into memory:
```json
{
  "status": "ok",
  "tensorflow_available": true
}
```

#### `POST /predict/solar?model={xgboost|lightgbm|catboost|lstm|ensemble}`
- **Tabular Models (`xgboost`, `lightgbm`, `catboost`):** Accepts single-row telemetry object.
- **Sequence Models (`lstm`, `ensemble`):** Accepts an array `rows` of the preceding 24 hourly records.

Example request for tabular:
```json
{
  "hour": 12,
  "day_of_week": 2,
  "day_of_year": 105,
  "month": 4,
  "household_id": 0,
  "temperature": 32.5,
  "cloud_cover": 15.0,
  "humidity": 45.0,
  "wind_speed": 4.2,
  "panel_capacity_kw": 5.0
}
```
Response:
```json
{
  "predicted_solar_generation_kwh": 3.42
}
```

#### `POST /predict/consumption?model={xgboost|lightgbm|catboost|lstm|ensemble}`
Returns the predicted household consumption in kWh for the requested slot.

---

## 7. Meta-Learner Stacking Details

The stacking ensemble combines predictions from XGBoost, LightGBM, and the LSTM. The Ridge meta-learner expects feature column names prefixed with `pred_`:
- `pred_xgboost`
- `pred_lightgbm`
- `pred_lstm`

`api/main.py` dynamically aligns prediction dictionaries against `ridge.feature_names_in_` to ensure strict dimensional and column alignment at runtime without sklearn schema mismatch errors.
