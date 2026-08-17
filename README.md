# Smart Energy Microgrid Trading — ML Pipeline


A minor project that builds an end-to-end machine learning pipeline for a
**smart energy microgrid**: synthetic household-level solar generation and
consumption data, exploratory data analysis, a multi-model regression
comparison, and a FastAPI service that serves live predictions. The
predicted solar generation and consumption values are the core inputs a
microgrid trading/scheduling layer would use to decide when a household is a
net energy producer (can sell/export) or a net consumer (must buy/import).


---


## 1. Problem Statement


A residential microgrid connects multiple households, each with rooftop
solar panels, to a shared local grid that allows energy trading between
households (surplus solar from one home can offset another's demand before
falling back to the utility grid). Efficient trading/scheduling decisions
require a **forecast** of two quantities per household, per hour:


1. **Solar generation (kWh)** — how much energy a household's panels will
   produce.
2. **Consumption (kWh)** — how much energy a household will use.


This project builds the forecasting layer: it generates a realistic
synthetic dataset (no public smart-meter dataset was available for this
scope), explores it, trains and compares several regression models for each
target, and exposes the best model of each as a REST prediction API.


## 2. Objectives


- Simulate realistic hourly solar generation and consumption patterns for
  multiple households over an extended period.
- Perform exploratory data analysis (EDA) to validate and visualize the
  patterns in the data.
- Train and fairly compare multiple regression algorithms per target using
  a time-respecting train/test split.
- Select and persist the best-performing model per target.
- Serve both models behind a documented, validated REST API suitable for
  integration into a larger trading/scheduling system.


## 3. Scope and Assumptions


- Data is **synthetic**, generated from parametric formulas designed to
  mimic real solar/load curves (bell-curve solar output peaking at midday,
  morning + evening consumption peaks, weekend effects) plus Gaussian noise
  — not measured from real smart meters.
- `household_id` and `day_of_week` are used as plain numeric/ordinal
  features rather than one-hot encoded, keeping the model input contract
  simple for the API. This is a reasonable simplification at this project's
  scale (10 households, 7 weekdays) and is an explicit design choice, not an
  oversight.
- The project focuses on the **forecasting layer** only; the actual
  peer-to-peer trading/matching logic (who sells to whom, at what price) is
  out of scope and would consume these predictions as an input.


---


## 4. System Architecture


```
                 ┌───────────────────────┐
                 │  generate_data.py     │   Synthetic hourly data
                 │  (data/)              │   generator
                 └──────────┬────────────┘
                            │  synthetic_energy_data.csv
                            ▼
                 ┌───────────────────────┐
                 │  eda.py               │   Missing-value check, describe(),
                 │  (notebooks/)         │   3 diagnostic plots
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌───────────────────────┐
                 │  train.py             │   Time-based split, trains 5
                 │                       │   regressors per target, compares
                 │                       │   RMSE/MSE/MAE/R2, saves best
                 └──────────┬────────────┘
                            │  best_solar_model.pkl
                            │  best_consumption_model.pkl
                            ▼
                 ┌───────────────────────┐
                 │  api/main.py          │   FastAPI service, models loaded
                 │  (FastAPI + Uvicorn)  │   once at startup
                 └───────────────────────┘
                    /health
                    /predict/solar
                    /predict/consumption
```


## 5. Project Structure


```
ml-service/
  data/
    generate_data.py            # synthetic data generator (Step 1)
    synthetic_energy_data.csv   # generated dataset (14,400 rows)
  notebooks/
    eda.py                      # EDA script (Step 2)
    plots/
      hourly_profile.png            # avg solar & consumption by hour
      correlation_heatmap.png       # correlation of numeric features
      consumption_boxplot_by_hour.png  # consumption variance/outliers by hour
  models/
    best_solar_model.pkl           # best model for solar_generation_kwh
    best_consumption_model.pkl     # best model for consumption_kwh
    solar_generation_kwh_comparison.csv   # full model comparison table
    consumption_kwh_comparison.csv        # full model comparison table
    *_scaler.pkl                   # only produced if SVR wins a target
  api/
    main.py                     # FastAPI app (Step 5)
  train.py                      # trains & compares 5 regressors (Steps 3-4)
  requirements.txt
  README.md
```


---


## 6. Dataset


### 6.1 Generation method
`data/generate_data.py` produces **hourly** records for **10 households**
over **60 days** (2026-01-01 to 2026-03-01), giving `10 × 24 × 60 = 14,400`
rows, with a fixed random seed (`42`) for reproducibility.


| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hour timestamp |
| `hour` | int (0–23) | Hour of day |
| `day_of_week` | int (0–6) | 0 = Monday … 6 = Sunday |
| `household_id` | int (0–9) | Household identifier |
| `temperature` | float | Ambient temperature (°C) |
| `solar_irradiance` | float (0–1) | Normalized solar irradiance |
| `panel_capacity_kw` | float | Fixed panel capacity per household (3–8 kW) |
| `solar_generation_kwh` | float | Target 1 — hourly solar energy generated |
| `consumption_kwh` | float | Target 2 — hourly household energy consumed |


### 6.2 Signal design
- **Temperature**: daily sinusoid (cooler at night, peak mid-afternoon) +
  a mild 60-day seasonal drift + Gaussian noise.
- **Solar irradiance**: Gaussian bell curve peaking at 1 PM, forced to ~0
  outside roughly 6 AM–7 PM (daylight window), clipped to `[0, 1]`.
- **Solar generation** = `panel_capacity_kw × solar_irradiance` + small
  Gaussian noise, clipped at 0 and forced to exactly 0 outside daylight
  hours (so nights are always 0, not just near-0 noise).
- **Consumption** = per-household baseline load + a Gaussian bump centered
  at 8 AM (morning peak) + a larger Gaussian bump centered at 7 PM
  (evening peak) + an extra daytime bump on weekends (Saturday/Sunday) +
  Gaussian noise, clipped at 0.
- Each household has a **fixed** panel capacity and baseline load drawn
  once (not per-row), so the same household behaves consistently across
  the whole 60-day window while still differing from its neighbors.


### 6.3 Summary statistics (generated dataset)


| Feature | Mean | Min | Max | Std |
|---|---|---|---|---|
| temperature (°C) | 18.01 | 4.65 | 31.38 | 6.12 |
| solar_irradiance | 0.31 | 0.00 | 1.00 | 0.35 |
| panel_capacity_kw | 5.98 | 3.47 | 7.88 | 1.45 |
| solar_generation_kwh | 1.83 | 0.00 | 7.99 | 2.24 |
| consumption_kwh | 1.21 | 0.00 | 3.24 | 0.63 |


No missing values in any column (validated in the EDA step).


---


## 7. Exploratory Data Analysis


`notebooks/eda.py` performs:


1. **Missing value check** — confirms 0 nulls across all 9 columns.
2. **`df.describe()`** — full numeric summary (table above).
3. **Hourly profile line plot** (`notebooks/plots/hourly_profile.png`) —
   average solar generation and consumption by hour of day. Confirms the
   expected shape: solar generation is 0 overnight, rises through the
   morning, peaks around 1 PM, and falls to 0 by evening; consumption
   shows a smaller morning bump (~8 AM) and a larger evening bump
   (~7 PM), with a low trough overnight.
4. **Correlation heatmap** (`notebooks/plots/correlation_heatmap.png`) —
   shows `solar_generation_kwh` correlates strongly with `solar_irradiance`
   and `panel_capacity_kw`, and weakly/negatively with `hour`-driven
   overnight zeros; `consumption_kwh` correlates with `hour` (peak-shaped,
   so linear correlation is moderate rather than strong) and `temperature`.
5. **Boxplot of consumption by hour** (`notebooks/plots/consumption_boxplot_by_hour.png`)
   — visualizes variance and outliers per hour; spread is visibly wider
   during the morning/evening peak hours than overnight, and weekend
   daytime hours show a distinct upward shift/outliers due to the added
   weekend bump.


---


## 8. Model Training & Comparison


### 8.1 Method
`train.py` trains **two independent pipelines** — one per target — using
five regression algorithms each:


- `LinearRegression`
- `SVR` (wrapped in a pipeline with `StandardScaler`, since SVR is
  scale-sensitive; tree-based models are trained on raw features since
  they don't need scaling)
- `DecisionTreeRegressor`
- `RandomForestRegressor` (`n_estimators=200`, `max_depth=15`,
  `min_samples_leaf=3` — regularized to control model size/overfitting,
  see §8.4)
- `XGBRegressor` (`n_estimators=200`)


**Feature sets:**


| Target | Features used |
|---|---|
| `solar_generation_kwh` | `hour`, `day_of_week`, `household_id`, `temperature`, `solar_irradiance`, `panel_capacity_kw` |
| `consumption_kwh` | `hour`, `day_of_week`, `household_id`, `temperature` |


### 8.2 Train/test split — time-based, not random
Because this is time series data, a random shuffle split would leak
future information into training (e.g. training on Tuesday 2 PM and
testing on Tuesday 1 PM of the *same* week is not a genuine
generalization test). Instead, the 60 unique calendar days are sorted,
and the **first 80% of days (48 days) form the training set**, the
**last 20% (12 days) form the held-out test set** — all households'
rows for a given day stay on the same side of the split.


### 8.3 Metrics
For each model: **RMSE**, **MSE**, **MAE**, and **R²** (the correct
term for a regression goodness-of-fit metric — not "accuracy", which
applies to classification). Results are sorted by R² descending.


### 8.4 Results


**Solar generation (`solar_generation_kwh`)**


| Rank | Model | RMSE | MSE | MAE | R² |
|---|---|---|---|---|---|
| 1 | **RandomForestRegressor** | 0.0448 | 0.0020 | 0.0269 | **0.9996** |
| 2 | XGBRegressor | 0.0489 | 0.0024 | 0.0296 | 0.9995 |
| 3 | DecisionTreeRegressor | 0.0583 | 0.0034 | 0.0351 | 0.9993 |
| 4 | SVR | 0.0639 | 0.0041 | 0.0507 | 0.9992 |
| 5 | LinearRegression | 0.5187 | 0.2690 | 0.4012 | 0.9465 |


**Consumption (`consumption_kwh`)**


| Rank | Model | RMSE | MSE | MAE | R² |
|---|---|---|---|---|---|
| 1 | **RandomForestRegressor** | 0.1597 | 0.0255 | 0.1275 | **0.9362** |
| 2 | XGBRegressor | 0.1646 | 0.0271 | 0.1320 | 0.9322 |
| 3 | DecisionTreeRegressor | 0.2179 | 0.0475 | 0.1732 | 0.8812 |
| 4 | SVR | 0.2648 | 0.0701 | 0.2083 | 0.8246 |
| 5 | LinearRegression | 0.5304 | 0.2813 | 0.4331 | 0.2962 |


**`RandomForestRegressor` was selected as the best model for both
targets** and pickled to `models/best_solar_model.pkl` and
`models/best_consumption_model.pkl`. No scaler artifact is produced in
this run since SVR did not win either target.


### 8.5 Interpretation
- **Solar generation** is almost perfectly predictable (R² ≈ 0.9996)
  because it's a near-deterministic function of `solar_irradiance` and
  `panel_capacity_kw` by construction — even `LinearRegression` reaches
  R² = 0.947, since the relationship is close to linear once irradiance
  is known. Tree ensembles edge out linear models by capturing the sharp
  day/night cutoff and the multiplicative interaction between irradiance
  and capacity.
- **Consumption** is meaningfully harder to predict (best R² ≈ 0.936)
  because it depends on non-linear, peak-shaped time-of-day effects and
  per-household baselines that a single linear model can't represent well
  (R² = 0.296) — this is exactly the kind of pattern tree ensembles are
  suited to, which is reflected in the large gap between linear and
  tree-based results.
- **Regularizing the RandomForestRegressor** (`max_depth=15,
  min_samples_leaf=3`) was a deliberate engineering decision: an
  unconstrained forest on the near-deterministic solar target grew trees
  ~27 levels deep with ~4,300 leaves each, producing a **122 MB** pickle
  file — impractical to commit to Git (GitHub blocks files over 100 MB)
  and unnecessarily slow to load. Regularizing brought the two model
  files down to **34 MB** and **51 MB** respectively while matching or
  slightly *improving* R² on the held-out test set (i.e. it reduced
  overfitting rather than trading accuracy for size).


---


## 9. Prediction API


`api/main.py` exposes the two best models via FastAPI. Both `.pkl` models
are loaded **once at application startup** (via a `lifespan` handler), not
on every request.


| Method & Path | Purpose |
|---|---|
| `GET /health` | Basic health check |
| `POST /predict/solar` | Predicts `solar_generation_kwh` |
| `POST /predict/consumption` | Predicts `consumption_kwh` |


All request bodies are validated with Pydantic (e.g. `hour` must be
0–23, `day_of_week` must be 0–6, `solar_irradiance` must be 0–1);
invalid input returns a `422` with a field-level error message.


### `POST /predict/solar`
Request:
```json
{
  "hour": 13,
  "day_of_week": 2,
  "household_id": 3,
  "temperature": 25.5,
  "solar_irradiance": 0.85,
  "panel_capacity_kw": 5.5
}
```
Response:
```json
{ "predicted_solar_generation_kwh": 4.43 }
```


### `POST /predict/consumption`
Request:
```json
{
  "hour": 19,
  "day_of_week": 5,
  "household_id": 3,
  "temperature": 22.0
}
```
Response:
```json
{ "predicted_consumption_kwh": 2.86 }
```


### `GET /health`
```json
{ "status": "ok" }
```


Interactive Swagger docs are auto-generated at `http://127.0.0.1:8000/docs`
once the server is running.


---


## 10. Tech Stack


| Layer | Tools |
|---|---|
| Data generation & manipulation | Python, NumPy, pandas |
| Visualization / EDA | Matplotlib, Seaborn |
| Modeling | scikit-learn (LinearRegression, SVR, DecisionTreeRegressor, RandomForestRegressor), XGBoost |
| Model persistence | joblib |
| API service | FastAPI, Pydantic, Uvicorn |


---


## 11. Setup & Run Instructions


From inside `ml-service/`:


```bash
python -3.13 -m venv venv


# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


pip install -r requirements.txt
```


Run in order (all commands from `ml-service/`, venv activated):


1. **Generate synthetic data**
   ```bash
   python data/generate_data.py
   ```
2. **Run EDA**
   ```bash
   python notebooks/eda.py
   ```
3. **Train and compare models**
   ```bash
   python train.py
   ```
4. **Start the API**
   ```bash
   uvicorn api.main:app --reload
   ```
   Then visit `http://127.0.0.1:8000/docs` or call the endpoints directly,
   e.g.:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict/solar \
     -H "Content-Type: application/json" \
     -d '{"hour":13,"day_of_week":2,"household_id":3,"temperature":25.5,"solar_irradiance":0.85,"panel_capacity_kw":5.5}'
   ```


---


## 12. Limitations & Future Scope


- **Synthetic data**: results demonstrate the pipeline and modeling
  approach, not real-world accuracy; validating against real smart-meter
  data would be the natural next step.
- **No weather forecast input**: `temperature` and `solar_irradiance` are
  treated as known inputs to the API, which in a real deployment would
  themselves need to be forecasted (e.g. from a weather API) 1–24 hours
  ahead.
- **No trading/matching logic**: this project produces the forecasts a
  microgrid trading algorithm would consume; the actual peer-to-peer
  matching, pricing, and settlement logic is a separate future module.
- **Categorical encoding**: `household_id` and `day_of_week` are used as
  raw numeric features rather than one-hot encoded; for a larger number
  of households this should be revisited (e.g. target/embedding encoding)
  to avoid implying a false ordinal relationship.
- **Single best model per target**: only the top model is persisted;
  ensembling the top 2–3 models per target could further improve
  robustness.


## 13. Conclusion


This project demonstrates a complete, working ML pipeline for a smart
microgrid forecasting problem: reproducible synthetic data generation,
thorough EDA, a rigorous time-aware multi-model comparison across five
regression algorithms, and a lightweight, validated REST API serving the
best model per target. `RandomForestRegressor` was the best-performing
model for both solar generation (R² = 0.9996) and consumption
(R² = 0.9362), and is what backs the deployed `/predict/solar` and
`/predict/consumption` endpoints.



