# Smart Energy Microgrid Trading — ML Pipeline


A Minor project that builds an end-to-end machine learning pipeline for a
**smart energy microgrid**: synthetic household-level solar generation and
consumption data, exploratory data analysis, a multi-model regression
comparison (Phase 1 baselines through Prophet/LightGBM/LSTM/stacking
ensemble), and a FastAPI service that serves live predictions from any of
them. The predicted solar generation and consumption values are the core
inputs a microgrid trading/scheduling layer would use to decide when a
household is a net energy producer (can sell/export) or a net consumer (must
buy/import).


---


## 1. Problem Statement


A residential microgrid connects multiple households, each with rooftop
solar panels, to a shared local grid that allows energy trading between
households (surplus solar from one home can offset another's demand before
falling back to the utility grid). Efficient trading/scheduling decisions
require a **forecast** of two quantities per household, per half-hour:


1. **Solar generation (kWh)** — how much energy a household's panels will
   produce.
2. **Consumption (kWh)** — how much energy a household will draw from the grid.


This project builds the forecasting layer: it preprocesses a real smart-meter
dataset, explores it, trains and compares a progression of models per target
(simple baselines through Prophet, SARIMAX, LightGBM, an LSTM, and a stacking
ensemble), and exposes them behind a REST prediction API.


## 2. Objectives


- Preprocess the real Ausgrid half-hourly dataset into a clean, feature-rich
  training dataset with correct Southern Hemisphere seasonality encoding,
  30-min weather interpolation, and per-household demand features.
- Perform exploratory data analysis (EDA) to validate and visualize the
  patterns in the data, including seasonal and household-level signals.
- Train and fairly compare a progression of models per target — Phase 1
  baselines (Linear/Tree/RandomForest/XGBoost), then Prophet, LightGBM,
  a pooled LSTM, and a stacking ensemble — using a time-respecting
  train/test (and train/validation/test) split throughout.
- Persist every model as a reusable artifact and combine every result into
  one honest, per-target comparison table and chart.
- Serve the models behind a documented, validated REST API suitable for
  integration into a larger trading/scheduling system.


## 3. Scope and Assumptions


- Data is the real **Ausgrid Solar Home Electricity Dataset** (300 households,
  July 2010 – June 2011, 30-min intervals). Households are **gross-metered**:
  all solar generation is exported to the grid and all consumption is imported
  from the grid, so `consumption_kwh = GC + CL` and `solar_gen_kwh = GG`.
- `household_id` is used as a numeric feature in all models (both solar and
  consumption), allowing models to learn per-household characteristics beyond
  what the aggregate features capture.
- Per-household demand level is represented by a continuous
  `hh_mean_consumption_kwh` feature (historical mean per interval) rather than
  a crude 3-bucket tier, retaining 100% of the variance the tier discards.
- `has_cl` (binary: controlled load present) is added as a feature for the
  consumption model to capture the nighttime CL spike pattern that affects 136
  of the 300 households.
- Prophet and SARIMAX are evaluated on one representative household per demand
  tertile (3 households each), not pooled. The LSTM and stacking ensemble pool
  all 300 households into a single model per target.
- The project focuses on the **forecasting layer** only; the actual
  peer-to-peer trading/matching logic is out of scope and would consume these
  predictions as an input.


---


## 4. System Architecture


```
               ┌───────────────────────────┐
               │  preprocess_ausgrid_data.py│   Real Ausgrid CSV → cleaned
               │  (data/)                  │   half-hourly CSV with cyclical
               └────────────┬──────────────┘   features, weather, HH stats
                            │  real_energy_data.csv
                            ▼
               ┌───────────────────────────┐
               │  eda.py                   │   Missing-value check, describe(),
               │  (notebooks/)             │   6 diagnostic plots
               └────────────┬──────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │  train.py                 │   Phase 1: time-based split, 4
               │                           │   regressors per target
               └────────────┬──────────────┘
                            │
                            ▼
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │ train_   │ train_   │ train_   │ train_   │ train_   │
   │ lgbm.py  │ prophet  │ sarimax  │ lstm.py  │ ensemble │
   │          │ .py      │ .py      │          │ .py      │
   └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
        │          │          │          │          │
        └──────────┴──────────┴──┬───────┴──────────┘
                                  │  models/metrics/*.json
                                  ▼
               ┌───────────────────────────┐
               │  model_comparison.py      │   Combines every model's metrics
               │  (results/)               │   into 2 tables + 2 bar charts
               └───────────────────────────┘

               ┌───────────────────────────┐
               │  api/main.py              │   FastAPI service, models loaded
               │  (FastAPI + Uvicorn)      │   once at startup; ?model= picks
               └───────────────────────────┘   which one serves the request
                  /health
                  /predict/solar
                  /predict/consumption
```


## 5. Project Structure


```
ml-service/
  data/
    ausgrid_dataset/            # raw Ausgrid CSVs + postcodes.csv
      weather_cache/            # cached Open-Meteo weather (auto-populated)
    preprocess_ausgrid_data.py  # Step 3: cleans + enriches Ausgrid data
    real_energy_data.csv        # preprocessed dataset (~5.26M rows)
  notebooks/
    eda.py                      # Step 4: EDA script
    plots/
      hourly_profile.png               # avg solar & consumption by half-hour
      correlation_heatmap.png          # correlation of numeric features
      consumption_boxplot_by_hour.png  # consumption spread by half-hour
      monthly_seasonal_profile.png     # monthly solar/consumption/temperature
      consumption_by_household.png     # per-HH mean consumption by has_cl
      radiation_distribution.png       # shortwave radiation distribution
  models/
    best_solar_model.pkl              # Phase 1 best model for solar_gen_kwh
    best_consumption_model.pkl        # Phase 1 best model for consumption_kwh
    xgboost_{solar,consumption}_model.pkl      # dedicated XGBoost artifacts (API)
    lightgbm_{solar,consumption}_model.pkl     # dedicated LightGBM artifacts (API)
    lstm_{solar,consumption}_model.keras       # pooled LSTM models
    lstm_{solar,consumption}_{target,features}_scaler.pkl  # LSTM MinMaxScalers
    ensemble_xgboost_{solar,consumption}_model.pkl   # ensemble's own base models
    ensemble_lightgbm_{solar,consumption}_model.pkl  # (only if LightGBM beat XGBoost)
    ridge_{solar,consumption}_meta.pkl        # stacking ensemble meta-learners (RidgeCV)
    ensemble_{solar,consumption}_config.json  # which base models the ensemble uses
    {target}_comparison.csv           # Phase 1 full model comparison table
    metrics/                          # one JSON per (target, model) - feeds results/
    train_lightgbm.py                 # LightGBM trainer
    train_prophet.py                  # Prophet benchmark (3 representative HH)
    train_sarimax.py                  # SARIMAX benchmark (3 representative HH)
    train_lstm.py                     # pooled LSTM trainer
    train_ensemble.py                 # XGBoost(+LightGBM)+LSTM stacking ensemble
  results/
    model_comparison.py          # combines models/metrics/*.json
    {solar,consumption}_model_comparison.csv   # full per-target comparison
    {solar,consumption}_r2_comparison.png      # R2 bar chart per target
  api/
    main.py                      # FastAPI app
  train.py                       # Phase 1: trains & compares 4 regressors
  requirements.txt
  steps.md                       # step-by-step guide to reproduce training
  README.md
```


---


## 6. Dataset


### 6.1 Source
**Ausgrid Solar Home Electricity Dataset** (publicly available from
ausgrid.com.au). This project uses the 2010–2011 year file: 300 solar
households in the greater Sydney area, metered at **30-minute** intervals,
July 2010 – June 2011.


### 6.2 Gross metering (key design decision)
All households are **gross-metered**: the solar system and household
circuits are on completely separate meters. This means:
- **GG** (Gross Generation) = all solar electricity generated → all exported
  to the grid. The household never self-consumes its own solar at the meter level.
- **GC** (General Consumption) = all electricity the household draws from the
  grid for general use.
- **CL** (Controlled Load) = off-peak tariff load drawn from the grid (hot
  water, pool pumps, etc.). Present in 136 of 300 households; the other 164
  have CL = 0 always.
- `consumption_kwh = GC + CL` — total grid draw, which equals total household
  consumption under gross metering.
- `solar_gen_kwh = GG`
- The net position for trading is `solar_gen_kwh − consumption_kwh` — positive
  means exporting, negative means importing. This is computed downstream from
  model outputs, not stored in the CSV.


### 6.3 Preprocessed CSV schema


**Identifier columns**

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Start of the 30-min interval (local EST/EDT) |
| `household_id` | int (0–299) | Dense factorized customer ID |

**Raw time columns (human inspection; not used as model inputs)**

| Column | Type | Description |
|---|---|---|
| `halfhour_idx` | int (0–47) | Half-hour-of-day index (0 = 00:00, 47 = 23:30) |
| `day_of_week` | int (0–6) | 0 = Monday, 6 = Sunday |
| `day_of_year` | int (1–365) | Day of year |

**Cyclical time encodings (model features)**

| Column | Formula | Notes |
|---|---|---|
| `halfhour_sin` | `sin(2π × halfhour_idx / 48)` | Time-of-day, cyclical |
| `halfhour_cos` | `cos(2π × halfhour_idx / 48)` | Time-of-day, cyclical |
| `dow_sin` | `sin(2π × day_of_week / 7)` | Day-of-week, cyclical |
| `dow_cos` | `cos(2π × day_of_week / 7)` | Day-of-week, cyclical |
| `doy_sin` | `sin(2π × (day_of_year − 355) / 365)` | **Southern Hemisphere** phase |
| `doy_cos` | `cos(2π × (day_of_year − 355) / 365)` | Peaks on Dec 21 (AU summer) |

**Household static features**

| Column | Type | Description |
|---|---|---|
| `panel_capacity_kw` | float | Rated panel capacity (kWp) |
| `hh_mean_consumption_kwh` | float | Household mean `consumption_kwh` per interval |
| `has_cl` | int (0/1) | 1 if household has any controlled load |

**Weather features (30-min interpolated from hourly Open-Meteo data)**

| Column | Type | Description |
|---|---|---|
| `temperature_c` | float | Air temperature (°C) |
| `shortwave_radiation_wm2` | float | Raw shortwave radiation (W/m²) — **not normalized** |

**Targets**

| Column | Formula | Description |
|---|---|---|
| `solar_gen_kwh` | GG | Gross solar generation per interval |
| `consumption_kwh` | GC + CL | Total household consumption per interval |


### 6.4 Feature sets used by models


| Target | Features |
|---|---|
| `solar_gen_kwh` | `halfhour_sin`, `halfhour_cos`, `doy_sin`, `doy_cos`, `household_id`, `panel_capacity_kw`, `shortwave_radiation_wm2`, `temperature_c` |
| `consumption_kwh` | `halfhour_sin`, `halfhour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`, `temperature_c`, `household_id`, `hh_mean_consumption_kwh`, `has_cl` |


---


## 7. Exploratory Data Analysis


`notebooks/eda.py` produces six plots saved to `notebooks/plots/`:

1. **Half-hourly profile** (`hourly_profile.png`) — average solar generation
   and consumption across the 48 half-hours of the day.
2. **Correlation heatmap** (`correlation_heatmap.png`) — numeric feature
   correlations, including the new cyclical features and `has_cl`.
3. **Consumption boxplot** (`consumption_boxplot_by_hour.png`) — variance
   and outliers per half-hour of day (every-other slot for readability).
4. **Monthly seasonal profile** (`monthly_seasonal_profile.png`) — monthly
   averages showing Australian seasonality (solar peaks Dec–Jan, troughs Jun–Jul).
5. **Per-household consumption scatter** (`consumption_by_household.png`) —
   each household's mean consumption, colored by `has_cl`, showing the
   continuous demand variation the old tier bins were coarsening.
6. **Radiation distribution** (`radiation_distribution.png`) — distribution
   of raw shortwave radiation W/m² values during daytime intervals, showing
   the full 0–1100+ W/m² range.


---


## 8. Phase 1: Baseline Model Training & Comparison


### 8.1 Method
`train.py` trains **two independent pipelines** — one per target — using
four regression algorithms each:

- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor` — `n_estimators=200`, `max_depth=20`,
  **`min_samples_leaf=20` for solar** (preserves the zero-boundary at
  dawn/dusk), **`min_samples_leaf=100` for consumption** (appropriate for
  the smoother demand signal at 5.26M rows).
- `XGBRegressor` — `n_estimators=500`, `max_depth=7`, `learning_rate=0.05`,
  `subsample=0.8`, `colsample_bytree=0.8` (centralized in `XGB_PARAMS`,
  shared with `train_lightgbm.py` and `train_ensemble.py`).

SVR is removed from this phase (it does not scale to 5.26M rows without a
prohibitively slow training time).


### 8.2 Train/test split — time-based, not random
The first 80% of calendar days → training set; the last 20% → held-out test
set. All households' rows for a given day stay on the same side. This
prevents future information leaking into training.


### 8.3 Metrics
For each model: **RMSE**, **MSE**, **MAE**, and **R²**. Every model's
metrics are written to `models/metrics/{target}__{model}.json` so
`results/model_comparison.py` can combine them across all phases without
re-running anything.


---


## 9. Advanced Models


### LightGBM (`models/train_lightgbm.py`)
`LGBMRegressor` trained on the same pooled 300-household data, same
time-based split, and same feature lists as XGBoost — a direct,
like-for-like comparison. Uses the same `LGBM_PARAMS` dict as
`train_ensemble.py` to prevent hyperparameter drift.


### Prophet (`models/train_prophet.py`)
Prophet is univariate per time series. Evaluated on **one representative
household per demand tertile** (derived dynamically from `hh_mean_consumption_kwh`
tertiles at runtime) — 3 households × 2 targets = 6 fits, reported as
individual rows. Deliberately fit as a pure `ds`/`y` univariate series
(no weather regressors) so it represents "seasonality decomposition of the raw
series" as an honest contrast to the causal-feature-driven tabular models.
If Prophet's Stan/`cmdstanpy` toolchain fails (blocked by a corporate proxy),
the script automatically falls back to a `statsmodels` SARIMAX model for
that row, labeled `SARIMAX_fallback_...`.


### SARIMAX (`models/train_sarimax.py`)
SARIMAX fills the unique "AR structure + exogenous causal features" slot in the
comparison:

| Model | AR structure | Exogenous features | Scope |
|---|---|---|---|
| Prophet | ✅ (Fourier seasonality) | ❌ (univariate) | 3 representative HH |
| **SARIMAX** | **✅ (AR/MA terms)** | **✅ weather + time** | **3 representative HH** |
| XGBoost/LGBM | ❌ | ✅ | All 300 HH |
| LSTM | ✅ (48-step window) | ✅ | All 300 HH |

Specification: `SARIMAX(1,0,1)(1,0,1,48)` — one AR/MA term, daily seasonal
period (48 half-hours), no differencing (both series are stationary).
Exogenous variables:
- Solar: `shortwave_radiation_wm2`, `temperature_c`, `doy_cos`
- Consumption: `temperature_c`, `dow_sin`, `dow_cos`, `has_cl`, `doy_cos`

Weekly and annual seasonality are handled via the exogenous regressors, not
additional seasonal ARIMA orders. Uses `method='powell'` to avoid a known
scipy version conflict with the default optimizer.


### LSTM (`models/train_lstm.py`)
A single **pooled** LSTM per target (all 300 households, with `household_id`
as an input feature channel) on **48-step (24-hour) sliding windows**.
All cyclical features are pre-computed in the CSV — the LSTM reads them
directly without re-computing. Architecture: `LSTM(128, return_sequences=True)
→ Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1)`,
trained with Adam (lr=0.001), up to 50 epochs with early stopping (patience=5).
Uses `batch_size=512` (increased from 64) so each epoch completes in a
reasonable time with ~5.2M sequences.


### Stacking Ensemble (`models/train_ensemble.py`)
Combines XGBoost + LSTM (+ LightGBM, only if it beat XGBoost in the Step 2
comparison) via a **`RidgeCV`** meta-learner with `alphas=[0.01, 0.1, 1.0, 10.0,
100.0]` — alpha is automatically tuned on the validation block, not hardcoded.
Uses a 3-way 70/10/20 train/validation/test day split (same 20% test block as
Phase 1, so results stay comparable). All base model hyperparameters are
imported from `train.XGB_PARAMS` and `train.LGBM_PARAMS` — no drift between
standalone and ensemble-base-model settings.


### Combined results (`results/model_comparison.py`)
Reads every `models/metrics/*.json` file and produces, per target: a full CSV
sorted by R² descending and an R² bar chart. Also prints an automated,
un-editorialized verdict on whether the ensemble actually beat the best
individual model for each target.


---


## 10. Prediction API


`api/main.py` exposes every model via FastAPI with a `?model=` query
param on `/predict/solar` and `/predict/consumption`:


| `?model=` value | Request body shape | Notes |
|---|---|---|
| `xgboost` | single row | dedicated XGBoost artifact from `train.py` |
| `lightgbm` | single row | from `train_lightgbm.py` |
| `lstm` | **48-row sequence** (past 48 half-hours, incl. observed target) | needs history |
| `ensemble` (default) | 48-row sequence | LSTM + XGBoost(+LightGBM) via RidgeCV meta-learner |
| `prophet` / `sarimax` | — | returns `400` — per-household benchmarks, not servable pooled models |


### Single-row request fields (v2)


**`POST /predict/solar`**
```json
{
  "halfhour_idx": 26,
  "day_of_year": 355,
  "day_of_week": 2,
  "household_id": 5,
  "panel_capacity_kw": 3.78,
  "shortwave_radiation_wm2": 650.0,
  "temperature_c": 28.5
}
```
Response:
```json
{ "predicted_solar_gen_kwh": 2.31 }
```


**`POST /predict/consumption`**
```json
{
  "halfhour_idx": 38,
  "day_of_year": 180,
  "day_of_week": 4,
  "household_id": 5,
  "temperature_c": 18.0,
  "hh_mean_consumption_kwh": 0.42,
  "has_cl": 1
}
```
Response:
```json
{ "predicted_consumption_kwh": 0.51 }
```

The cyclical sin/cos features (`halfhour_sin/cos`, `dow_sin/cos`, `doy_sin/cos`)
are computed automatically by the API from the raw index fields — callers do not
need to pre-compute them.


### `GET /health`
```json
{ "status": "ok" }
```

Interactive Swagger docs auto-generated at `http://127.0.0.1:8000/docs`.


---


## 11. Tech Stack


| Layer | Tools |
|---|---|
| Data preprocessing | Python, NumPy, pandas, requests (Open-Meteo API) |
| Visualization / EDA | Matplotlib, Seaborn |
| Modeling — Phase 1 & tabular | scikit-learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, Ridge), XGBoost, LightGBM |
| Modeling — time series | Prophet (with a statsmodels SARIMAX fallback) |
| Modeling — deep learning | TensorFlow/Keras (LSTM) |
| Model persistence | joblib, Keras `.keras` format |
| API service | FastAPI, Pydantic, Uvicorn |


---


## 12. Setup & Run Instructions


From inside `ml-service/`:

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Run in order (all commands from `ml-service/`, venv activated):

1. **Preprocess Ausgrid data** *(needs internet connection first time)*
   ```bash
   python data/preprocess_ausgrid_data.py
   ```
2. **Run EDA**
   ```bash
   python notebooks/eda.py
   ```
3. **Phase 1: train and compare baseline models**
   ```bash
   python train.py
   ```
4. **Advanced models** (`train_ensemble.py` depends on LightGBM + LSTM having already run)
   ```bash
   python models/train_lightgbm.py
   python models/train_prophet.py
   python models/train_sarimax.py
   python models/train_lstm.py
   python models/train_ensemble.py
   ```
5. **Combine every result into one comparison table + chart**
   ```bash
   python results/model_comparison.py
   ```
6. **Start the API**
   ```bash
   uvicorn api.main:app --reload
   ```
   Then visit `http://127.0.0.1:8000/docs`.

See `steps.md` for the full step-by-step guide including GPU setup, expected
outputs at each step, and troubleshooting.


---


## 13. Limitations & Future Scope


- **Weather forecast input**: `temperature_c` and `shortwave_radiation_wm2`
  are treated as known inputs to the API, which in a real deployment would
  themselves need to be forecast (e.g. from a weather API) 1–24 hours ahead.
- **No trading/matching logic**: this project produces the forecasts a
  microgrid trading algorithm would consume; the actual peer-to-peer
  matching, pricing, and settlement logic is a separate future module.
- **Prophet/SARIMAX scope**: evaluated on 3 of 300 households — diagnostic
  benchmarks, not servable pooled models. A full per-household or hierarchical
  setup is future work.
- **LSTM/ensemble API scoping**: requiring a 48-row history payload for
  `?model=lstm`/`?model=ensemble` (vs. a single row for `xgboost`/`lightgbm`)
  is a pragmatic demo-project simplification. A real deployment would likely
  maintain server-side rolling history per household.
- **Single year**: only the 2010–2011 year is used. Enabling the 2011–2012 and
  2012–2013 files (already referenced but commented out in `preprocess_ausgrid_data.py`)
  would give 3× more training data and improve generalization, especially for
  the LSTM and ensemble.


## 14. Conclusion


This project demonstrates a complete, working ML pipeline for a smart
microgrid forecasting problem: a real smart-meter dataset with correct Southern
Hemisphere preprocessing, thorough EDA, a time-aware Phase 1 baseline comparison,
and a progression into LightGBM, Prophet, SARIMAX, a pooled LSTM, and a stacking
ensemble — combined into one honest, automatically-generated comparison table and
chart per target, served behind a REST API that lets a caller pick which model
answers each request.
