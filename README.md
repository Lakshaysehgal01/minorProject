# Smart Energy Microgrid Trading — ML Pipeline


A minor project that builds an end-to-end machine learning pipeline for a
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
require a **forecast** of two quantities per household, per hour:


1. **Solar generation (kWh)** — how much energy a household's panels will
   produce.
2. **Consumption (kWh)** — how much energy a household will use.


This project builds the forecasting layer: it generates a realistic
synthetic dataset (no public smart-meter dataset was available for this
scope), explores it, trains and compares a progression of models per target
(simple baselines through Prophet, LightGBM, an LSTM, and a stacking
ensemble), and exposes them behind a REST prediction API.


## 2. Objectives


- Simulate realistic hourly solar generation and consumption patterns for
  multiple households over a full year, including genuine seasonal (winter
  vs. summer) variation and differentiated household profiles.
- Perform exploratory data analysis (EDA) to validate and visualize the
  patterns in the data, including the added seasonality and household-tier
  signals.
- Train and fairly compare a progression of models per target — Phase 1
  baselines (Linear/SVR/Tree/RandomForest/XGBoost), then Prophet, LightGBM,
  a pooled LSTM, and a stacking ensemble — using a time-respecting
  train/test (and train/validation/test) split throughout.
- Persist every model as a reusable artifact and combine every result into
  one honest, per-target comparison table and chart.
- Serve the models behind a documented, validated REST API suitable for
  integration into a larger trading/scheduling system.


## 3. Scope and Assumptions


- Data is **synthetic**, generated from parametric formulas designed to
  mimic real solar/load curves (bell-curve solar output peaking at midday
  with a seasonally-varying window and intensity, morning + evening
  consumption peaks scaled by household size, weekend effects) plus
  Gaussian noise — not measured from real smart meters.
- `household_id` and `day_of_week` are used as plain numeric/ordinal
  features rather than one-hot encoded, keeping the model input contract
  simple for the API — a reasonable simplification at this project's scale
  (25 households, 7 weekdays), and an explicit design choice, not an
  oversight. The household's *size* (small/medium/large usage tier) is
  captured separately via the genuinely ordinal `consumption_profile_encoded`
  feature, which avoids the "false ordinal relationship" concern that a raw
  `household_id` alone would have.
- Prophet is evaluated on a small representative sample of households (one
  per tier), not pooled across all 25 — see §8bis. The LSTM and stacking
  ensemble, by contrast, pool all 25 households into a single model per
  target.
- The project focuses on the **forecasting layer** only; the actual
  peer-to-peer trading/matching logic (who sells to whom, at what price) is
  out of scope and would consume these predictions as an input.


---


## 4. System Architecture


```
                 ┌───────────────────────┐
                 │  generate_data.py     │   Synthetic hourly data
                 │  (data/)              │   generator (25 households, 365 days)
                 └──────────┬────────────┘
                            │  energy_data.csv
                            ▼
                 ┌───────────────────────┐
                 │  eda.py               │   Missing-value check, describe(),
                 │  (notebooks/)         │   5 diagnostic plots
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌───────────────────────┐
                 │  train.py             │   Phase 1: time-based split, 5
                 │                       │   regressors per target
                 └──────────┬────────────┘
                            │
                            ▼
     ┌──────────────┬──────────────┬──────────────┬──────────────┐
     │ train_        │ train_       │ train_        │ train_       │
     │ lightgbm.py   │ prophet.py   │ lstm.py       │ ensemble.py  │
     │ (models/)     │ (models/)    │ (models/)     │ (models/)    │
     └──────┬────────┴──────┬───────┴──────┬────────┴──────┬───────┘
            │               │              │               │
            └───────────────┴──────┬───────┴───────────────┘
                                    │  models/metrics/*.json
                                    ▼
                 ┌───────────────────────┐
                 │  model_comparison.py  │   Combines every model's metrics
                 │  (results/)           │   into 2 tables + 2 bar charts
                 └───────────────────────┘


                 ┌───────────────────────┐
                 │  api/main.py          │   FastAPI service, models loaded
                 │  (FastAPI + Uvicorn)  │   once at startup; ?model= picks
                 └───────────────────────┘   which one serves the request
                    /health
                    /predict/solar
                    /predict/consumption
```


## 5. Project Structure


```
ml-service/
  data/
    fetch_weather.py            # downloads & caches real Delhi weather (Step 1)
    generate_data.py            # synthetic data generator, pvlib solar physics (Step 1)
    weather_cache/
      delhi_weather_2024.csv    # cached real hourly Delhi weather (Open-Meteo)
    energy_data.csv             # generated dataset (219,000 rows)
  notebooks/
    eda.py                      # EDA script (Step 2)
    plots/
      hourly_profile.png               # avg solar & consumption by hour
      correlation_heatmap.png          # correlation of numeric features
      consumption_boxplot_by_hour.png  # consumption variance/outliers by hour
      monthly_seasonal_profile.png     # NEW: monthly solar/consumption/temperature
      consumption_boxplot_by_profile.png # NEW: consumption by household tier
  models/
    trained_models/
      best_solar_model.pkl              # Phase 1 best model for solar_generation_kwh
      best_consumption_model.pkl        # Phase 1 best model for consumption_kwh
      xgboost_{solar,consumption}_model.pkl      # dedicated XGBoost artifacts (API)
      lightgbm_{solar,consumption}_model.pkl     # dedicated LightGBM artifacts (API)
      catboost_{solar,consumption}_model.pkl     # dedicated CatBoost artifacts (API)
      lstm_{solar,consumption}_model.keras       # pooled LSTM models
      lstm_{solar,consumption}_{target,features}_scaler.pkl  # LSTM MinMaxScalers
      ensemble_xgboost_{solar,consumption}_model.pkl    # ensemble's own base models
      ensemble_lightgbm_{solar,consumption}_model.pkl   # (only if LightGBM beat XGBoost)
      ensemble_catboost_{solar,consumption}_model.pkl   # (only if CatBoost beat XGBoost)
      ridge_{solar,consumption}_meta.pkl        # stacking ensemble meta-learners
      ensemble_{solar,consumption}_config.json  # which base models the ensemble uses
      {target}_comparison.csv           # Phase 1 full model comparison table
    metrics/                          # one JSON per (target, model) - feeds results/
    train_lightgbm.py                 # Step 2: LightGBM (Phase 2)
    train_catboost.py                 # Step 2b: CatBoost (Phase 2)
    train_prophet.py                  # Step 1: Prophet, 3-household benchmark
    train_lstm.py                     # Step 3: pooled LSTM
    train_ensemble.py                 # Step 4: XGBoost(+LightGBM/CatBoost)+LSTM stack
  results/
    model_comparison.py          # Step 5: combines models/metrics/*.json
    {solar,consumption}_model_comparison.csv   # full per-target comparison
    {solar,consumption}_r2_comparison.png      # R2 bar chart per target
  api/
    main.py                      # FastAPI app (Step 6)
  train.py                       # Phase 1: trains & compares 5 regressors
  requirements.txt
  README.md
```


---


## 6. Dataset


### 6.1 Why this dataset was reworked
The first version of this generator used smooth, noise-free sinusoids for
temperature/irradiance and fed `solar_irradiance` directly to the models as
a feature. Since `solar_generation_kwh ≈ panel_capacity_kw × solar_irradiance`
is almost an exact linear function of that feature, **every** model -
Linear Regression through the full stacking ensemble - landed at
R² ≈ 0.997-0.9997 for solar, with almost no separation between model
families. That's not a modeling win, it's a data problem: there was no real
unexplained variance left for any model to fail on. See §8bis's "Results"
note and §12 for the full before/after.


### 6.2 Generation method (current)
`data/generate_data.py` produces **hourly** records for **25 households**
over a **full year** (2026-01-01 to 2026-12-31, 365 days), giving
`25 × 24 × 365 = 219,000` rows, with a fixed random seed (`42`) for
reproducibility. Two things now come from the real world instead of a
formula:


- **Real weather**: `data/fetch_weather.py` downloads one real calendar
  year (2024) of hourly Delhi weather - temperature, humidity, cloud cover,
  wind speed, and GHI/DNI/DHI solar irradiance components - from the free,
  keyless [Open-Meteo Historical Weather API](https://open-meteo.com/) and
  caches it to `data/weather_cache/delhi_weather_2024.csv`. `generate_data.py`
  calls this automatically the first time it's needed and reuses the cache
  afterward, so it only needs network access once. (On a network with a
  TLS-inspecting corporate proxy, this fetch can fail with an SSL
  certificate error - if so, run it once from an unrestricted network, or
  on the training machine, and copy the resulting cache file over.)
- **Real solar physics**: [pvlib](https://pvlib-python.readthedocs.io/)
  converts that real irradiance into each household's actual panel output -
  plane-of-array irradiance for a randomized tilt/azimuth, ambient- and
  wind-driven cell temperature (`pvlib.temperature.faiman`), PVWatts DC
  power with per-household dust/soiling loss and mild annual panel
  degradation, and PVWatts inverter conversion (which clips output once DC
  power exceeds the undersized inverter's rating) - instead of a hand-typed
  formula.


Indian public holidays (`holidays.India`) and several outlier/anomaly
generators (inverter faults, equipment-outage days, blackout blocks biased
toward the hottest hours, appliance-surge spikes) round out the realism -
see §6.4.


| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hour timestamp |
| `hour` | int (0–23) | Hour of day |
| `day_of_week` | int (0–6) | 0 = Monday … 6 = Sunday |
| `day_of_year` | int (1–365) | Day of year |
| `month` | int (1–12) | Calendar month |
| `is_holiday` | int (0–1) | Indian public holiday flag |
| `household_id` | int (0–24) | Household identifier |
| `consumption_profile` | string | Household usage tier: "small"/"medium"/"large" |
| `consumption_profile_encoded` | int (0–2) | Ordinal encoding of the tier above |
| `temperature` | float | Real Delhi ambient temperature (°C), + per-household sensor bias/noise |
| `cloud_cover` | float (0–100) | Real cloud cover (%) |
| `humidity` | float (0–100) | Real relative humidity (%) |
| `wind_speed` | float | Real wind speed |
| `solar_irradiance` | float | Plane-of-array irradiance, reference/EDA only - **not** a model feature (see §6.1) |
| `panel_capacity_kw` | float | Fixed panel capacity per household (2–12 kW) |
| `solar_generation_kwh` | float | Target 1 — hourly solar energy generated (pvlib output) |
| `consumption_kwh` | float | Target 2 — hourly household energy consumed |


### 6.3 Signal design
- **Solar generation**: real GHI/DNI/DHI → pvlib plane-of-array irradiance
  (per-household random tilt 25-31°, azimuth 170-190°) → PVWatts DC power
  (with per-household soiling decay + reset "cleanings" every ~15-30 days,
  and ~0.5%/year degradation) → PVWatts inverter conversion (DC:AC ≈ 1.11,
  so sunny-day output visibly clips instead of scaling forever).
- **Consumption** = the same per-household baseline + morning/evening
  Gaussian peaks + weekend daytime bump as before, **plus** a
  cooling-degree-hour AC term driven by real temperature/humidity (a rough
  heat-index adjustment), an extra evening bump on real Indian holidays, and
  an AR(1) (day-to-day correlated, not i.i.d.) daily "mood" noise term on
  top of the existing hourly noise.
- Households are split into 3 tiers (10 small / 10 medium / 5 large,
  shuffled across `household_id`), each with its own baseline-load range
  (`small` 0.2–0.4 kW, `medium` 0.4–0.7 kW, `large` 0.7–1.2 kW). Panel
  capacity, tilt, azimuth, soiling schedule, and temperature-sensor bias are
  all drawn independently per household and fixed for the whole year.


### 6.4 Outliers & anomalies (extreme values, not missing data)
By design, anomalies show up as real zero/spike readings rather than `NaN`
gaps, so no imputation code is needed anywhere downstream:
- **Solar**: rare (~0.3%) inverter-fault zeros and rare (~0.2%) measurement
  spikes during daylight hours, plus 1-3 full-day equipment-outage days per
  household per year.
- **Consumption**: 2-5 multi-hour blackout blocks per household per year,
  biased toward the hottest hours (realistic Delhi grid strain during heat
  waves), plus rare single-hour appliance-surge spikes.


Summary statistics and EDA plots depend on the real weather cache, which
this repo doesn't ship with (see §6.2) - run `python notebooks/eda.py`
after generating the data to see the real distributions.


---


## 7. Exploratory Data Analysis


`notebooks/eda.py` performs:


1. **Missing value check** — confirms 0 nulls across all 17 columns (the
   dataset has no missing-data gaps by design - see §6.4).
2. **`df.describe()`** — full numeric summary.
3. **Hourly profile line plot** (`hourly_profile.png`) — average solar
   generation and consumption by hour of day.
4. **Correlation heatmap** (`correlation_heatmap.png`) — numeric feature
   correlations, including the new `consumption_profile_encoded` column.
5. **Boxplot of consumption by hour** (`consumption_boxplot_by_hour.png`) —
   variance/outliers per hour.
6. **Monthly seasonal profile** (`monthly_seasonal_profile.png`, new) —
   monthly-average solar generation, consumption, and temperature, to
   verify the added annual seasonality actually shows up (solar/temperature
   should visibly peak mid-year and trough in winter).
7. **Consumption by household tier** (`consumption_boxplot_by_profile.png`,
   new) — verifies small < medium < large consumption ordering holds.


---


## 8. Phase 1: Baseline Model Training & Comparison


### 8.1 Method
`train.py` trains **two independent pipelines** — one per target — using
five regression algorithms each:


- `LinearRegression`
- `SVR` (wrapped in a pipeline with `StandardScaler`, since SVR is
  scale-sensitive; tree-based models are trained on raw features since
  they don't need scaling)
- `DecisionTreeRegressor`
- `RandomForestRegressor` (`n_estimators=200`, `max_depth=15`,
  `min_samples_leaf=25` — regularized to control model size/overfitting;
  bumped up from `3` when the dataset scaled to 219,000 rows, see §8.4)
- `XGBRegressor` (`n_estimators=200`)


**Feature sets:**


| Target | Features used |
|---|---|
| `solar_generation_kwh` | `hour`, `day_of_week`, `day_of_year`, `month`, `household_id`, `temperature`, `cloud_cover`, `humidity`, `wind_speed`, `panel_capacity_kw` |
| `consumption_kwh` | `hour`, `day_of_week`, `household_id`, `temperature`, `humidity`, `is_holiday`, `consumption_profile_encoded` |


`solar_irradiance` is deliberately **not** a feature - see §6.1 for why.


### 8.2 Train/test split — time-based, not random
Because this is time series data, a random shuffle split would leak
future information into training. Instead, the 365 unique calendar days
are sorted, and the **first 80% of days (292 days) form the training
set**, the **last 20% (73 days, Oct 20 – Dec 31) form the held-out test
set** — all households' rows for a given day stay on the same side of the
split. Note this test window sits in the autumn→winter transition and
never overlaps the training data's winter solstice, so models must
partially extrapolate into a seasonal regime they saw less of — expected
behavior, not a bug.


### 8.3 Metrics
For each model: **RMSE**, **MSE**, **MAE**, and **R²**. Every model's
metrics are additionally written to `models/metrics/{target}__{model}.json`
so `results/model_comparison.py` can combine them with every later phase's
results without re-running anything.


### 8.4 Results


> **These results predate the §6.1 data-realism rework and are now stale** -
> they were measured on the old smooth-sinusoid dataset with `solar_irradiance`
> as a feature, which is exactly the setup that produced unrealistic
> R² ≈ 0.999 scores across the board. They're kept here as a historical
> record of that finding; re-run `train.py` on the regenerated
> `data/energy_data.csv` (real weather + pvlib physics) to get current
> numbers, which should show a genuinely harder problem and real
> separation between model families. See §12 for the recommended
> hyperparameter-tuning follow-up once that re-run is done.


Trained end-to-end on the full 219,000-row dataset (see `ml-service/STEPS.md`
for the exact reproduction steps):


**Solar generation (`solar_generation_kwh`)**


| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | **RandomForestRegressor** | 0.0345 | 0.0164 | **0.99972** |
| 2 | XGBRegressor | 0.0361 | 0.0186 | 0.99970 |
| 3 | DecisionTreeRegressor | 0.0498 | 0.0243 | 0.99942 |
| 4 | SVR | 0.1710 | 0.1245 | 0.99319 |
| 5 | LinearRegression | 0.6949 | 0.5690 | 0.88754 |


**Consumption (`consumption_kwh`)**


| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | **RandomForestRegressor** | 0.1515 | 0.1210 | **0.95832** |
| 2 | XGBRegressor | 0.1526 | 0.1218 | 0.95773 |
| 3 | DecisionTreeRegressor | 0.2154 | 0.1722 | 0.91575 |
| 4 | LinearRegression | 0.5403 | 0.4279 | 0.46990 |


*SVR was not run to completion for consumption in this training run — its
training time was prohibitive at this row count (~175K training rows), a
tradeoff explicitly flagged during design (see §12). Solar's own SVR result
(R²=0.993, well below every tree/boosting model) suggests it would likely
have ranked similarly low for consumption, so its absence doesn't change
which model wins.*


`RandomForestRegressor` was selected as the best Phase 1 model for **both**
targets, pickled to `models/trained_models/best_solar_model.pkl` and
`models/trained_models/best_consumption_model.pkl`. As a sizing note: an earlier
*unregularized* RandomForest on the original 10-household dataset produced
a 122 MB pickle; scaling to 219,000 rows and bumping `min_samples_leaf`
from 3 to 25 kept both pickles in a similar, practical size range on this
run.


### 8.5 Interpretation (historical - see the §8.4 callout)
- **Solar generation** reached R²=0.99972 with RandomForest, and every
  tree/boosting model landed within a hair of it — confirms it's a
  near-deterministic function of `solar_irradiance` and `panel_capacity_kw`,
  both given directly as features. Even `LinearRegression` reached
  R²=0.888, since the relationship is close to linear once irradiance is
  known.
- **Consumption** topped out lower (R²=0.958, RandomForest) — confirms
  it's meaningfully harder, since it depends on non-linear, peak-shaped
  time-of-day effects and per-household/tier baselines that
  `LinearRegression` (R²=0.470) can't represent well — exactly the kind of
  pattern tree ensembles and gradient boosting are suited to.


---


## 8bis. Advanced Models: Prophet, LightGBM, CatBoost, LSTM, Stacking Ensemble


Building on Phase 1, five more model scripts live in `models/`:


### Prophet (`models/train_prophet.py`)
Prophet is univariate per time series, so instead of pooling all 25
households it's evaluated on **one representative household per
consumption tier** (the lowest `household_id` in each tier: small=4,
medium=0, large=3) — 3 households × 2 targets = 6 fits, reported as
individual rows (not averaged, since tiers differ enough in scale that an
average wouldn't represent any real household). Deliberately fit as a pure
`ds`/`y` univariate series (no `temperature`/`solar_irradiance` regressor)
so it represents "seasonality decomposition of the raw series" as an
honest, weaker contrast to the causal-feature-driven tabular models — not
a serving-ready pooled model. If Prophet's install/first run fails (its
Stan/`cmdstanpy` toolchain can be blocked by a corporate proxy), the script
automatically falls back to a `statsmodels` SARIMAX model (daily
seasonality only) for that row, labeled `SARIMAX_fallback_...`.


### LightGBM (`models/train_lightgbm.py`)
An `LGBMRegressor(n_estimators=200)` trained on the same pooled
25-household data, same time-based split, and same feature lists as
XGBoost — a direct, like-for-like comparison point.


### CatBoost (`models/train_catboost.py`)
A `CatBoostRegressor(n_estimators=200, verbose=False)`, otherwise identical
in setup to LightGBM — same pooled data, split, and feature lists, so all
three boosters (XGBoost, LightGBM, CatBoost) are directly comparable.


### LSTM (`models/train_lstm.py`)
A single **pooled** LSTM per target (all 25 households, with
`household_id` as an input feature, rather than 25 separate models) on
24-hour sliding windows. `hour`/`day_of_week`/`day_of_year` are cyclically
(sin/cos) encoded for this model specifically. Sequences are built
per-household and dropped if they'd straddle a train/validation/test
boundary, so no window crosses a household or split boundary. All numeric
inputs (not just the target) are MinMax-scaled, fit on the training split
only. Architecture: `LSTM(64, return_sequences=True) → Dropout(0.2) →
LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)`, trained with Adam
(lr=0.001), up to 50 epochs with early stopping (patience=5).


### Stacking Ensemble (`models/train_ensemble.py`)
Combines XGBoost + LSTM (+ every candidate booster — currently LightGBM and
CatBoost — that individually beat XGBoost in its own Phase 2 comparison,
not capped at one) via a `Ridge` meta-learner. Uses a 3-way 70/10/20 train
/validation/test day split (same 73-day test block as Phase 1, so results
stay comparable): base models are fit on the 70% train block, their
validation-set predictions train the Ridge meta-learner, and the full
stack is evaluated on the untouched 20% test block. Because the LSTM has
no predictions for the first 24 hours of each household's period, all base
model predictions are aligned via an inner join on `(household_id,
timestamp)` rather than positional slicing. Prophet is **not** a base
model here — mixing a 3-household benchmark into a 25-household pooled
ensemble would shrink the ensemble's own evaluation down to just those 3
households, defeating the point of a pooled comparison. Adding a future
booster only needs a new entry in `train_ensemble.py`'s `CANDIDATE_BOOSTERS`
dict, not new branching logic.


### Combined results (`results/model_comparison.py`)
Reads every `models/metrics/*.json` file (Phase 1 + LightGBM + CatBoost +
Prophet + LSTM + Ensemble) and produces, per target: a full CSV
(`results/{target}_model_comparison.csv`, sorted by R² descending,
including all 3 Prophet-household rows) and an R² bar chart
(`results/{target}_r2_comparison.png`, with the 3 Prophet rows collapsed
to one median-R² bar for readability). It also prints an automated,
un-editorialized verdict on whether the ensemble actually beat the best
individual model for each target — reported either way.


### Results (full pipeline, all phases combined)


> **Stale, same as §8.4**: the tables and honesty-check analysis below were
> measured on the old smooth-sinusoid, `solar_irradiance`-as-a-feature
> dataset (see §6.1) - that's *why* every model here clusters so tightly
> near R²=0.999 for solar. They also predate CatBoost. Re-run
> `train_lightgbm.py`, `train_catboost.py`, `train_prophet.py`,
> `train_lstm.py`, `train_ensemble.py`, and `results/model_comparison.py`
> on the training machine against the regenerated `data/energy_data.csv` to
> get a current, meaningful combined table.


**Solar generation (`solar_generation_kwh`)**


| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | RandomForestRegressor | 0.0345 | 0.0164 | 0.99972 |
| 2 | LGBMRegressor | 0.0360 | 0.0185 | 0.99970 |
| 3 | XGBRegressor | 0.0361 | 0.0186 | 0.99970 |
| 4 | **Ensemble** | 0.0379 | 0.0197 | 0.99966 |
| 5 | DecisionTreeRegressor | 0.0498 | 0.0243 | 0.99942 |
| 6 | SVR | 0.1710 | 0.1245 | 0.99319 |
| 7 | LSTM | 0.2795 | 0.2052 | 0.98175 |
| 8 | LinearRegression | 0.6949 | 0.5690 | 0.88754 |
| 9 | Prophet (household 4, small) | 0.3483 | 0.2509 | 0.74844 |
| 10 | Prophet (household 0, medium) | 1.1598 | 0.8647 | 0.74773 |
| 11 | Prophet (household 3, large) | 1.3193 | 1.0232 | 0.61132 |


**Consumption (`consumption_kwh`)**


| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | LGBMRegressor | 0.1502 | 0.1200 | 0.95902 |
| 2 | **Ensemble** | 0.1504 | 0.1201 | 0.95896 |
| 3 | RandomForestRegressor | 0.1515 | 0.1210 | 0.95832 |
| 4 | XGBRegressor | 0.1526 | 0.1218 | 0.95773 |
| 5 | LSTM | 0.1703 | 0.1354 | 0.94738 |
| 6 | DecisionTreeRegressor | 0.2154 | 0.1722 | 0.91575 |
| 7 | Prophet (household 3, large) | 0.3212 | 0.2586 | 0.87460 |
| 8 | Prophet (household 0, medium) | 0.3890 | 0.3299 | 0.59819 |
| 9 | Prophet (household 4, small) | 0.3141 | 0.2518 | 0.49196 |
| 10 | LinearRegression | 0.5403 | 0.4279 | 0.46990 |


**Honest finding: the ensemble does not beat the single best individual
model on either target** — a real, reported-as-is outcome from the
automated honesty check in `results/model_comparison.py`, not a bug:
- Solar: Ensemble R²=0.99966 vs. RandomForest's 0.99972 (best individual) —
  a ~0.00006 gap.
- Consumption: Ensemble R²=0.95896 vs. LightGBM's 0.95902 (best
  individual) — again ~0.00006.


Both share the same cause. LightGBM narrowly beat XGBoost in Phase 2 on
*both* targets, so per the base-model selection rule in this section, the
ensemble's base models were XGBoost + LightGBM + LSTM for both targets.
LSTM is meaningfully weaker than the tree-based models on both targets
(R²=0.982 vs. ~0.9997 for solar; R²=0.947 vs. ~0.958 for consumption), and
the Ridge meta-learner had only a 37-day validation block to learn how
much to trust each base model — not quite enough data to fully suppress
the weaker LSTM's contribution, so the blend ends up very slightly worse
than just using the strongest tree model alone. This is a common, valid
outcome for stacking ensembles: they don't automatically improve on the
best base learner, especially when one base model underperforms by a wide
margin and validation data is limited.


A secondary finding worth noting: **LightGBM edges out Phase 1's
RandomForest for consumption specifically** (0.95902 vs. 0.95832), though
close enough to be within noise; RandomForest remains the outright best
for solar even after every later phase was added.


---


## 9. Prediction API


`api/main.py` exposes every model via FastAPI with a `?model=` query
param on `/predict/solar` and `/predict/consumption`:


| `?model=` value | Request body shape | Notes |
|---|---|---|
| `xgboost` | single row (as before) | dedicated XGBoost artifact from `train.py` |
| `lightgbm` | single row | from `train_lightgbm.py` |
| `catboost` | single row | from `train_catboost.py` |
| `lstm` | **24-row sequence** (past 24 hours, incl. the actual observed target for each) | needs history to feed the LSTM |
| `ensemble` (default) | 24-row sequence | LSTM + XGBoost(+LightGBM/CatBoost) blended via the Ridge meta-learner |
| `prophet` | — | returns `400` — it's a 3-household benchmark, not a servable pooled model |


The `lstm`/`ensemble` sequence requirement is a deliberate, documented
scoping decision for this demo project: a single row has no 24-hour
history to feed the LSTM, so those two options intentionally use a
different request shape than `xgboost`/`lightgbm`/`catboost`. Any
model/artifact not present on the running server (e.g. TensorFlow not
installed, or a
training script hasn't been run yet) makes only the affected `?model=`
choice return `503` — it doesn't stop the service from starting.


All request bodies are validated with Pydantic (e.g. `hour` must be
0–23, `day_of_week` must be 0–6, `cloud_cover`/`humidity` must be 0–100);
invalid input returns a `422` with a field-level error message.


### `POST /predict/solar?model=xgboost`
Request:
```json
{
  "hour": 13,
  "day_of_week": 2,
  "day_of_year": 180,
  "month": 6,
  "household_id": 3,
  "temperature": 28.5,
  "cloud_cover": 20.0,
  "humidity": 45.0,
  "wind_speed": 3.2,
  "panel_capacity_kw": 8.0
}
```
Response:
```json
{ "predicted_solar_generation_kwh": 6.13 }
```


### `POST /predict/consumption?model=lightgbm`
Request:
```json
{
  "hour": 19,
  "day_of_week": 2,
  "household_id": 3,
  "temperature": 15.0,
  "humidity": 55.0,
  "is_holiday": 0,
  "consumption_profile_encoded": 2
}
```
Response:
```json
{ "predicted_consumption_kwh": 3.67 }
```


### `POST /predict/solar?model=ensemble`
Request (abbreviated — 24 rows total, oldest to newest, each including the
actual observed `solar_generation_kwh` for that past hour):
```json
{
  "rows": [
    { "hour": 0, "day_of_week": 2, "day_of_year": 180, "month": 6, "household_id": 3,
      "temperature": 20.1, "cloud_cover": 15.0, "humidity": 40.0, "wind_speed": 2.1,
      "panel_capacity_kw": 8.0, "solar_generation_kwh": 0.0 },
    { "...": "23 more hourly rows" }
  ]
}
```
Response:
```json
{ "predicted_solar_generation_kwh": 5.98 }
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
| Real-world data sourcing | Open-Meteo Historical Weather API (`requests`), pvlib (solar panel physics), `holidays` (Indian public holidays) |
| Visualization / EDA | Matplotlib, Seaborn |
| Modeling — Phase 1 & tabular | scikit-learn (LinearRegression, SVR, DecisionTreeRegressor, RandomForestRegressor, Ridge), XGBoost, LightGBM, CatBoost |
| Modeling — time series | Prophet (with a statsmodels SARIMAX fallback) |
| Modeling — deep learning | TensorFlow/Keras (LSTM) |
| Model persistence | joblib, Keras `.keras` format |
| API service | FastAPI, Pydantic, Uvicorn |


---


## 11. Setup & Run Instructions


From inside `ml-service/`:


```bash
python -m venv venv


# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


pip install -r requirements.txt
```


Run in order (all commands from `ml-service/`, venv activated) — later
steps depend on earlier ones' output:


1. **Generate synthetic data**
   ```bash
   python data/generate_data.py
   ```
   The first run downloads and caches one real year of Delhi weather
   (needs internet access once - see §6.2's corporate-proxy note if it
   fails with an SSL error); every run after that reuses the cached file.
2. **Run EDA**
   ```bash
   python notebooks/eda.py
   ```
3. **Phase 1: train and compare baseline models**
   ```bash
   python train.py
   ```
4. **Advanced models** (each depends on step 3's data/CSV; `train_ensemble.py`
   additionally depends on steps 5 and 6 having already run)
   ```bash
   python models/train_lightgbm.py
   python models/train_prophet.py
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
   Then visit `http://127.0.0.1:8000/docs` or call the endpoints directly,
   e.g.:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict/solar?model=xgboost" \
     -H "Content-Type: application/json" \
     -d '{"hour":13,"day_of_week":2,"day_of_year":180,"month":6,"household_id":3,"temperature":28.5,"cloud_cover":20.0,"humidity":45.0,"wind_speed":3.2,"panel_capacity_kw":8.0}'
   ```


---


## 12. Limitations & Future Scope


- **Synthetic data, real weather**: the target values (`solar_generation_kwh`,
  `consumption_kwh`) are still simulated, but they're now driven by a real
  year of Delhi weather + pvlib panel physics rather than a smooth formula
  (§6.1) - closer to real-world accuracy, but validating against real
  smart-meter data would still be the natural next step.
- **No weather forecast input**: `temperature`/`cloud_cover`/`humidity`/
  `wind_speed` are treated as known inputs to the API, which in a real
  deployment would themselves need to be forecasted (e.g. from a weather
  API) 1–24 hours ahead, and forecast values carry their own error that
  this dataset doesn't currently simulate.
- **Results tables need re-running**: §8.4's and §8bis's results tables
  were measured before the §6.1 data-realism rework and are now stale -
  see those sections' callouts. Re-run every `train_*.py` script and
  `results/model_comparison.py` on the training machine against the
  regenerated `data/energy_data.csv`.
- **Recommended next step - hyperparameter tuning**: now that the problem
  is genuinely harder, default hyperparameters (e.g. `n_estimators=200`
  across the board) will likely stop being trivially near-perfect. A
  tuning pass (`GridSearchCV`/Optuna over XGBoost/LightGBM/CatBoost) should
  show real, meaningful differentiation between models for the first time -
  not implemented here, but worth doing once the re-run above is complete.
- **No trading/matching logic**: this project produces the forecasts a
  microgrid trading algorithm would consume; the actual peer-to-peer
  matching, pricing, and settlement logic is a separate future module.
- **Prophet's limited scope**: evaluated on 3 of 25 households and
  univariate-only, so it's a diagnostic benchmark, not a servable pooled
  model; a full per-household or hierarchical Prophet setup is future work.
- **LSTM/ensemble API scoping**: requiring a 24-row history payload for
  `?model=lstm`/`?model=ensemble` (vs. a single row for `xgboost`/
  `lightgbm`) is a pragmatic demo-project simplification, not a unified
  production API — a real deployment would likely maintain server-side
  rolling history per household instead of requiring the caller to supply it.
- **SVR not evaluated for consumption**: this run's SVR training was
  prohibitively slow for consumption at ~175K training rows and wasn't
  completed; based on solar's SVR result (well below the tree/boosting
  models there), it's unlikely to have changed the ranking, but it's a gap
  in the full comparison table worth closing with a subsampled SVR run if
  a complete picture is needed.
- **Stacking ensemble didn't beat the best individual model**: see §8bis —
  reported honestly rather than presented as an improvement; a stronger
  ensemble would likely need either a better-performing LSTM or a larger
  validation block for the meta-learner to properly downweight the
  weaker base model.


## 13. Conclusion


This project demonstrates a complete, working ML pipeline for a smart
microgrid forecasting problem: reproducible synthetic data generation with
genuine annual seasonality and differentiated household profiles, thorough
EDA, a time-aware Phase 1 baseline comparison, and a progression into
Prophet, LightGBM, a pooled LSTM, and a stacking ensemble — combined into
one honest, automatically-generated comparison table and chart per target,
served behind a REST API that lets a caller pick which model answers each
request.


**Best model per target**: `RandomForestRegressor` (solar) and
`LGBMRegressor` (consumption) won the most recent full run, with the
stacking ensemble not surpassing the best individual model on either
target - reported as-is rather than framed as a win, consistent with this
project's "report the real numbers honestly" principle throughout. Those
specific numbers (§8.4, §8bis) predate the real-weather/pvlib data-realism
rework in §6.1 and need re-measuring; the rework exists precisely because
the old numbers (R² ≈ 0.999 for solar, near-identical scores across very
different model families) were an artifact of a too-easy, noise-free
dataset rather than a genuine modeling result.





