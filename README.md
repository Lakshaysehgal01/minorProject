# Smart Energy Microgrid Trading

A minor project that builds a complete **peer-to-peer solar trading system
for a smart energy microgrid** (25 households, hourly), from forecasting to
settlement to a dashboard:

1. **ML forecasting** (`ml-service/`): synthetic household solar generation
   and consumption driven by real Delhi weather, a multi-model comparison
   (baselines through Prophet/LightGBM/CatBoost/LSTM/stacking ensemble), and a
   FastAPI service that serves live predictions.
2. **Trading and settlement** (`contracts/`, `backend/`): every simulated hour
   the backend asks the ML service for each household's solar and consumption,
   works out who has surplus and who has a deficit, matches them, and settles
   each trade on a local Ethereum chain (Hardhat) through a Solidity contract.
3. **Dashboard** (`frontend/`): a React app that runs the simulation day by
   day and shows predictions vs actuals, trades, on-chain balances, savings
   and the model comparison.

What is **real** and what is **simulated** is spelled out in
[§17](#17-what-is-real-and-what-is-simulated). In short: the ML predictions
and the on-chain settlement are real; the smart-meter readings, the weather
replay and the tariffs are simulated.

## Contents

- [Quick start (full stack)](#quick-start-full-stack)
- [Overview: architecture and repository layout](#overview-architecture-and-repository-layout)
- **Part I: ML pipeline**: [§1 Problem](#1-problem-statement) ·
  [§2 Objectives](#2-objectives) · [§3 Scope](#3-scope-and-assumptions) ·
  [§4 Architecture](#4-system-architecture) · [§5 Structure](#5-project-structure) ·
  [§6 Dataset](#6-dataset) · [§7 EDA](#7-exploratory-data-analysis) ·
  [§8 Phase 1](#8-phase-1-baseline-model-training--comparison) ·
  [§8bis Advanced models](#8bis-advanced-models-prophet-lightgbm-catboost-lstm-stacking-ensemble) ·
  [§9 Prediction API](#9-prediction-api) · [§10 Tech stack](#10-tech-stack) ·
  [§11 ML setup](#11-setup--run-instructions) · [§12 Limitations](#12-limitations--future-scope) ·
  [§13 Conclusion](#13-conclusion)
- **Part II: trading layer**: [§14 Smart contract](#14-smart-contract-contracts) ·
  [§15 Backend](#15-backend-backend) · [§16 Frontend](#16-frontend-frontend) ·
  [§17 Real vs simulated](#17-what-is-real-and-what-is-simulated) ·
  [§18 Example run](#18-example-run-and-what-it-shows) ·
  [§19 Testing](#19-testing) · [§20 Trading-layer limitations](#20-trading-layer-limitations--future-scope)

---

## Quick start (full stack)

**Prerequisites:** Node.js 20+ (tested on 24) and npm; Python 3 for
`ml-service/` (a `venv` with `requirements.txt` installed, see
[§11](#11-setup--run-instructions)); a MongoDB you can reach (local install
or an Atlas connection string). Everything below works on Windows, macOS and
Linux; all npm scripts are cross-platform.

**1. Generate the ML artifacts (once).** From `ml-service/` with its venv
active (they are gitignored, so a fresh clone has none):

```bash
python data/generate_data.py          # dataset (needs internet once for weather, see §6.2)
python train.py                       # Phase 1 models, incl. the XGBoost artifact
python models/train_lightgbm.py       # default model served to the backend
python models/train_catboost.py
python results/model_comparison.py    # optional: feeds the dashboard's Model Results tab
```

**2. Install JavaScript dependencies and configure** (from the repo root):

```bash
npm run install:all
copy backend\.env.example backend\.env      # macOS/Linux: cp backend/.env.example backend/.env
```

Edit `backend/.env` and set `MONGO_URI`. Every other value has a working
default. `frontend/.env` is optional (see [§16](#16-frontend-frontend)).

**3. Start everything with one command** (from the repo root):

```bash
npm run start:all
```

`scripts/start-all.js` (Node, so it works on Windows, macOS and Linux) starts
the Hardhat node, waits for it, deploys the contract, then starts the ML
service, the backend and the dashboard, each in order and waiting until the
previous one answers. All output appears in one terminal with a colour-coded
tag per service (`[chain]`, `[deploy]`, `[ml]`, `[backend]`, `[frontend]`), and
**Ctrl+C stops every service**. It checks first that `backend/.env` and the
ML virtualenv (`ml-service/venv`) exist, and refuses to start if any of ports
8545, 8000, 4000 or 5173 is already in use (stop the old processes first).

| Command | Effect |
|---|---|
| `npm run start:all` | start the full stack |
| `npm run start:all -- --reset` | also call `POST /simulate/reset` once the backend is up. Use this when MongoDB still holds trades from an earlier run, because a fresh chain starts empty. It clears the simulation data in MongoDB |
| `npm run start:all -- --mock` | start only the dashboard with generated demo data (no backend, ML service, chain or MongoDB needed) |

Without `--reset`, the launcher prints a warning if it detects that MongoDB
is ahead of the fresh chain.

*Prefer to start the services by hand* (for example to see one service's logs
on its own)? Run these in separate terminals, in this order:

| # | What | Command | Port |
|---|---|---|---|
| 1 | Local blockchain (Hardhat node, in-memory) | `npm run chain` | 8545 |
| 2 | Deploy the contract and register 25 households (one-shot) | `npm run deploy` | n/a |
| 3 | ML prediction service (from `ml-service/`, venv active) | `uvicorn api.main:app --port 8000` | 8000 |
| 4 | Backend API (MongoDB must be reachable) | `npm run dev` | 4000 |
| 5 | Dashboard | `npm run dev:frontend` | 5173 |

**4. Open http://localhost:5173 and click "Run Next Day".** One click
simulates 24 hourly slots (about 30 s): predictions, matching, and on-chain
settlement. Or run it without the UI: `curl -X POST http://localhost:4000/simulate/day`.

**5. Verify the whole stack** with `npm run smoke`. It runs one simulated day
against the live services and checks ledger invariants (see [§19](#19-testing)).

Other scripts: `npm run start:all`, `npm run test:contracts`, `npm run test:backend`,
`npm run build` (backend), `npm run build:frontend`, and
`npm run dev:frontend:mock` to demo the UI with no backend at all.

> **Chain restarts:** the Hardhat node keeps its state in memory only. If you
> restart it, run `npm run deploy` again and then `POST /simulate/reset` (or
> just use `npm run start:all -- --reset`): the
> backend refuses to run while its database is ahead of the chain, and
> `GET /health` says so.
>
> **Ports:** the backend listens on 4000 (not 5000) and the dashboard on 5173,
> the only origin the backend's CORS allows.

---

## Overview: architecture and repository layout

```
                       ┌──────────────────────────────────────────┐
                       │ ml-service/  (Python, FastAPI :8000)     │
                       │  synthetic data → train → predict API    │
                       └───────────────▲──────────────────────────┘
                                       │ POST /predict/solar|consumption
                                       │ (single row per household, per hour)
┌────────────────────────┐   ┌─────────┴────────────────────────────┐   ┌──────────────────────────┐
│ energy_data.csv        │──▶│ backend/  (Node + TypeScript :4000)  │──▶│ contracts/  (Hardhat     │
│ replayed hour by hour  │   │  data replay → ML client → matching  │   │ node :8545)              │
│ (Oct 20 – Dec 31 2026) │   │  → on-chain settlement → savings     │   │ EnergyTrading.sol ledger │
└────────────────────────┘   │  MongoDB: predictions, trades, cursor│   └──────────────────────────┘
                             └─────────▲────────────────────────────┘
                                       │ REST (JSON)
                             ┌─────────┴────────────────────────────┐
                             │ frontend/  (React + Vite :5173)      │
                             │  Dashboard · Model Results           │
                             └──────────────────────────────────────┘
```

```
.
├── ml-service/        ML pipeline + FastAPI prediction service (Part I, §1–§13)
│   ├── data/          weather download + synthetic data generator
│   ├── models/        training scripts; trained_models/ and metrics/ are gitignored
│   ├── results/       model comparison CSVs/charts (gitignored except the script)
│   ├── api/main.py    FastAPI app
│   ├── train.py       Phase 1 baselines
│   └── STEPS.md       exact reproduction steps
├── contracts/         Hardhat project: EnergyTrading.sol, tests, deploy script (§14)
├── backend/           Express + TypeScript + Mongoose + ethers API (§15)
│   └── README.md      backend details: API table, definitions, behaviour notes
├── frontend/          React + TypeScript + Vite + Tailwind + recharts dashboard (§16)
│   └── README.md      frontend details: env vars, mock mode, design notes
├── scripts/
│   └── start-all.js   one-command launcher for the whole stack (npm run start:all)
├── package.json       root scripts that delegate to contracts/, backend/, frontend/
└── README.md          this file
```

Trading runs **hourly**: a solar surplus at noon cannot offset an evening
deficit without storage. "Daily" is only an aggregation of 24 hourly ticks.

---

# Part I: ML forecasting pipeline

> Sections 1–13 describe the forecasting layer (`ml-service/`). The
> trading layer that consumes it is described in Part II (§14–§20).

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
- Part I (this section through §13) covers the **forecasting layer** only.
  The peer-to-peer matching, settlement and dashboard that consume these
  predictions are implemented separately in Part II (§14–§20): a flat-price
  greedy matcher, an on-chain ledger and a React dashboard, with no
  dynamic pricing, no storage and no authentication.

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

The backend in Part II is this API's main client: it calls
`/predict/solar` and `/predict/consumption` once per household per simulated
hour. By default (`ML_MODEL=auto`) the backend picks, per target and at
startup, the servable model with the highest R² in the comparison CSVs
(`xgboost`, `lightgbm`, `catboost`, `lstm` or `ensemble`); there is no
fallback, so startup fails with a clear message if the CSVs are missing.

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
  STEPS.md                       # exact reproduction steps
```

(This tree covers `ml-service/` only. The repository's other folders,
`contracts/`, `backend/` and `frontend/`, are shown in the
[overview](#overview-architecture-and-repository-layout).)

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

#### Current results (regenerated dataset: real weather + pvlib physics)

Re-measured with `train.py`, `models/train_lightgbm.py` and
`models/train_catboost.py` on the regenerated `data/energy_data.csv`
(219,000 rows, same 73-day test block), then combined by
`results/model_comparison.py`. This is the problem the §6.1 rework intended:
genuinely harder, with real separation between model families.

**Solar generation (`solar_generation_kwh`)**

| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | **LGBMRegressor** | 0.4608 | 0.2050 | **0.9578** |
| 2 | XGBRegressor | 0.5199 | 0.2498 | 0.9462 |
| 3 | CatBoostRegressor | 0.5321 | 0.2925 | 0.9437 |
| 4 | RandomForestRegressor | 0.5507 | 0.2472 | 0.9397 |
| 5 | DecisionTreeRegressor | 0.8136 | 0.3021 | 0.8684 |
| 6 | LinearRegression | 2.8127 | 2.1011 | −0.5731 |

**Consumption (`consumption_kwh`)**

| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | **LGBMRegressor** | 0.2467 | 0.1918 | **0.9102** |
| 2 | CatBoostRegressor | 0.2473 | 0.1924 | 0.9098 |
| 3 | XGBRegressor | 0.2544 | 0.1978 | 0.9045 |
| 4 | RandomForestRegressor | 0.2561 | 0.1982 | 0.9032 |
| 5 | DecisionTreeRegressor | 0.3563 | 0.2776 | 0.8127 |
| 6 | LinearRegression | 0.6115 | 0.4933 | 0.4484 |

What this shows:
- **LightGBM is the best model for both targets** (solar R² 0.958,
  consumption R² 0.910), which is why it is the backend's default
  (and what the backend's `ML_MODEL=auto` selects from these tables). For consumption, LightGBM, CatBoost and XGBoost are
  within about 0.006 R² of each other, so that ranking is close to noise.
- **Solar dropped from R² ≈ 0.9997 to ≈ 0.94–0.96** once `solar_irradiance`
  stopped being a feature and the weather became real. The old near-perfect
  score was an artifact of the easy setup, not a modelling result.
- **`LinearRegression` fails on solar (R² = −0.57, worse than predicting the
  mean).** Output is a non-linear function of hour and cloud cover, and the
  test window (Oct 20 – Dec 31) moves into a seasonal regime the training
  window saw less of (§8.2).
- **Not re-run for this table:** SVR (commented out in `train.py`, it is too
  slow at this row count), LSTM, the stacking ensemble and Prophet. The
  Prophet rows that `results/model_comparison.py` still prints come from a
  metrics file written on the earlier dataset and are therefore stale; re-run
  `models/train_prophet.py` before quoting them.

#### Historical results (old smooth-sinusoid dataset, kept for the record)

> The two tables below were measured on the old dataset with
> `solar_irradiance` as a feature, the setup that produced the unrealistic
> R² ≈ 0.999 scores across the board. They are kept as a record of that
> finding and are **not** comparable with the current results above.

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

### 8.5 Interpretation (historical, old dataset; current takeaways are in §8.4)
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

> **Partly stale.** LightGBM and CatBoost have been re-measured on the
> regenerated dataset (see the current results in §8.4). The combined tables
> and honesty-check analysis below, which include Prophet, LSTM and the
> stacking ensemble, were measured on the old smooth-sinusoid,
> `solar_irradiance`-as-a-feature dataset (see §6.1), which is *why* every
> model here clusters so tightly near R²=0.999 for solar. They also predate
> CatBoost. To get a current combined table, re-run `train_prophet.py`,
> `train_lstm.py`, `train_ensemble.py` and `results/model_comparison.py`
> against the regenerated `data/energy_data.csv`.

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
| Smart contract (Part II) | Solidity 0.8.24, Hardhat 2 (local node only), OpenZeppelin `Ownable`, chai tests |
| Trading backend (Part II) | Node.js, TypeScript, Express 5, Mongoose 8 (MongoDB), ethers 6, axios, zod, Vitest |
| Dashboard (Part II) | React 19, TypeScript, Vite, Tailwind CSS 4, recharts, axios |

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
   If Python's `requests` fails on a TLS-inspecting proxy, a workaround that
   worked on a corporate Windows machine was to download the same
   Open-Meteo URL with the OS-native `curl.exe` (which trusts the Windows
   certificate store) and save it as `data/weather_cache/delhi_weather_2024.csv`
   in the format `fetch_weather.py` writes (columns `time`,
   `temperature_2m`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`,
   `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`,
   `precipitation`); `generate_data.py` then never needs the network.
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
- **Results tables only partly re-measured**: §8.4 now has current numbers
  for the Phase 1 models, LightGBM and CatBoost, but SVR, Prophet, LSTM and
  the stacking ensemble have not been re-run on the regenerated dataset, so
  §8.4's historical tables and §8bis's combined tables are still stale (see
  their callouts). Re-run `train_prophet.py`, `train_lstm.py`,
  `train_ensemble.py` and `results/model_comparison.py`.
- **Recommended next step - hyperparameter tuning**: now that the problem
  is genuinely harder, default hyperparameters (e.g. `n_estimators=200`
  across the board) will likely stop being trivially near-perfect. A
  tuning pass (`GridSearchCV`/Optuna over XGBoost/LightGBM/CatBoost) should
  show real, meaningful differentiation between models for the first time -
  not implemented here, but worth doing once the re-run above is complete.
- **Trading logic lives in Part II, with its own limits**: the peer-to-peer
  matching, settlement and dashboard now exist (§14–§16), but pricing is a
  flat tariff and there is no storage; see §20.
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

**Best model per target**: on the regenerated real-weather dataset,
`LGBMRegressor` is best for both solar (R² 0.958) and consumption (R² 0.910);
see the current results in §8.4. The earlier full run, where
`RandomForestRegressor` won solar and the stacking ensemble failed to beat
the best single model, was measured on the old noise-free dataset and is kept
only as history. The rework exists precisely because the old numbers (R² ≈
0.999 for solar, near-identical scores across very different model families)
were an artifact of a too-easy dataset rather than a genuine modeling result.
Prophet, LSTM and the ensemble still need re-running on the new data before
any claim about them is current, reported honestly rather than carried over.

The forecasts feed a working trading layer (Part II): a Solidity contract
settling peer-to-peer trades on a local chain, a backend that replays the
test period hour by hour, and a React dashboard.

---

# Part II: trading layer (contracts, backend, dashboard)

## 14. Smart Contract (`contracts/`)

`contracts/contracts/EnergyTrading.sol` is the on-chain ledger for peer-to-peer
trades (Solidity 0.8.24, OpenZeppelin v5 `Ownable`). It runs on a **local
Hardhat node only**: there is no testnet or mainnet deployment.

| Function | Who | What it does |
|---|---|---|
| `registerHousehold(householdId, wallet)` | owner | registers a household; rejects a duplicate id or the zero address |
| `executeTrade(sellerId, buyerId, kWhMilli, pricePerKwhMilli, slotTimestamp)` | owner | settles one trade; returns the on-chain `tradeId` and emits `TradeExecuted` |
| `getEnergyBalance(id)`, `getTokenBalance(id)` | anyone | signed balances |
| `getTradeCount()`, `getTrade(tradeId)` | anyone | the stored trade history (`getTrade` is bounds-checked) |

How a trade settles: the call requires both households to be registered,
`seller != buyer`, `kWhMilli > 0` and `pricePerKwhMilli > 0`. Then the
seller's energy balance goes down and the buyer's goes up by `kWhMilli`, the
buyer's token balance goes down and the seller's goes up by the cost, and a
`Trade` record is stored.

- **Units:** energy is in milli-kWh (kWh × 1000) and prices in milli-credits
  per kWh, so there are no decimals on-chain. `cost = kWhMilli *
  pricePerKwhMilli / 1000` (integer division, rounds down). The backend
  computes the same integer formula.
- **Balances are signed ledger entries**, not an ERC-20 and not escrow: a net
  buyer's credit balance can go negative, and the energy balances across all
  households always sum to zero. "Credits" are settlement units, not real money.
- **`slotTimestamp` is an addition to the original 4-argument design.** The
  block timestamp is wall-clock time, which says nothing about *which
  simulated hour* a trade belongs to. Storing the simulated slot on-chain
  makes the chain a self-contained record and lets the backend recover an
  interrupted trade by looking for its `TradeExecuted` event.
- **Trusted operator:** only the owner (the account that deployed the
  contract, which is the backend's signer) can register households or
  execute trades. Households do not sign anything.

Tooling is plain JavaScript on Hardhat 2 (`hardhat.config.js`). The `hardhat`
network is configured with **26 accounts** (Hardhat's default is 20, but 25
households are registered). `npm run deploy` deploys the contract, registers
households 0–24 with the first 25 accounts, and writes the address, chain id,
deploy block and ABI to `backend/src/contracts/EnergyTrading.json` (generated,
gitignored).

## 15. Backend (`backend/`)

An Express 5 + TypeScript API (plain exported functions, no classes) with
MongoDB (Mongoose) for state and ethers.js for the chain. It replays the test
period, **Oct 20 – Dec 31 2026 = 73 days = 1,752 hourly slots**, starting at
2026-10-20 00:00. The full reference (API table, definitions, behaviour notes)
is in [`backend/README.md`](backend/README.md); the essentials:

**One hourly slot (`runSlot`):**

1. Read the 25 households' rows for that hour from `energy_data.csv`: weather
   inputs, panel size, tier, holiday flag, and the *actual* meter values.
2. Call the ML service for each household's solar and consumption (single-row
   models, 5 s timeout, predictions clamped to ≥ 0) and save them as
   `Prediction` documents.
3. `surplus = predicted solar − predicted consumption`. A household with a
   surplus above `MIN_KWH` (default 0.5) is a seller, one with a deficit above
   it is a buyer. Greedily match the largest remaining seller with the largest
   remaining buyer, trading the smaller of the two quantities, at the flat
   `P2P_PRICE_PER_KWH`. Ties go to the lower household id, so runs are
   deterministic.
4. Settle each match on-chain **one at a time** (nonce safety). A trade that
   reverts is recorded as `failed` and the slot continues.
5. Advance the persisted cursor (last, so a crash cannot skip a slot).

**Endpoints** (default `http://localhost:4000`):

| Method | Route | Purpose |
|---|---|---|
| POST | `/simulate/tick` | run one hourly slot |
| POST | `/simulate/day` | run until the next 00:00 (24 slots from midnight) and return the day summary |
| POST | `/simulate/run?hours=N` | run N slots (1–168) |
| POST | `/simulate/reset` | clear predictions, trades and the cursor |
| GET | `/daily-summary?date=YYYY-MM-DD` | one day's totals, % deficit met P2P, savings, prediction MAE |
| GET | `/households` | tier, panel kW, on-chain energy and token balance |
| GET | `/trades?page&limit&date&status` | trades, newest first |
| GET | `/trades/household/:id` | one household's trades |
| GET | `/predictions/:householdId?date=` | hourly predicted vs actual |
| GET | `/forecast/:householdId?date=` | predictions only (no actuals), from the dataset's weather columns |
| GET | `/stats` | all-time totals, savings, MAE, simulation progress |
| GET | `/results/models` | the model comparison tables from `ml-service/results` |
| GET | `/models/active` | the model in use per target (also returned as `models` by `/forecast`) |
| GET | `/health` | MongoDB, ML service (a real probe prediction), chain, and chain/DB sync |

Errors return `{ "error": "..." }` with 400 (validation), 404 (no data yet),
409 (busy / replay finished / chain and DB out of sync), 502 (bad ML
response) or 503 (a dependency is down).

**Robustness:**
- A slot is **idempotent**: predictions are upserted, a trade is written as
  `pending` before it is sent, and a `pending` trade left by an interrupted
  run is reconciled against on-chain `TradeExecuted` events (matched on seller,
  buyer and `slotTimestamp`) instead of being re-sent.
- Only **one simulation request runs at a time**; a concurrent one gets `409`.
- The Hardhat node is in-memory while Mongo persists, so before running the
  backend checks that the contract exists and that the chain holds at least as
  many trades as the database has confirmed; otherwise it answers `409` and
  tells you to redeploy and `POST /simulate/reset`.
- Dataset timestamps are naive local time and are parsed as **UTC**
  everywhere, so results do not depend on the host's time zone.

**Configuration** (`backend/.env`, from `backend/.env.example`; validated with
zod at startup):

| Variable | Default | Meaning |
|---|---|---|
| `MONGO_URI` | none, required | MongoDB connection string |
| `PORT` | 4000 | backend port |
| `ML_SERVICE_URL` | `http://localhost:8000` | prediction service |
| `ML_MODEL` | `auto` | `auto` picks, per target and at startup, the highest-R² model among `xgboost`/`lightgbm`/`catboost`/`lstm`/`ensemble` listed in `ML_RESULTS_DIR`'s `{solar,consumption}_model_comparison.csv`; it errors if the CSVs are missing. Set a model name to pin one; `ML_MODEL_SOLAR` / `ML_MODEL_CONSUMPTION` override per target |
| `DATA_CSV_PATH`, `ML_RESULTS_DIR` | `../ml-service/...` | dataset and comparison CSVs (relative to `backend/`) |
| `RPC_URL`, `OWNER_PRIVATE_KEY`, `CONTRACT_ADDRESS` | local node, Hardhat account #0 | chain access; the key is publicly known, so never use it on a real network; `CONTRACT_ADDRESS` overrides the address in the deploy JSON |
| `P2P_PRICE_PER_KWH`, `GRID_PRICE`, `FEEDIN_PRICE` | 6, 8, 3 | tariffs; must satisfy `GRID > P2P > FEEDIN > 0` |
| `MIN_KWH` | 0.5 | seller/buyer eligibility threshold |
| `CORS_ORIGIN` | `http://localhost:5173` | allowed browser origin(s) |

**Savings definitions.** With `GRID_PRICE` (buy from the utility) >
`P2P_PRICE_PER_KWH` > `FEEDIN_PRICE` (sell to the utility):
- Per confirmed trade the buyer saves `(GRID − P2P) × kWh` and the seller gains
  `(P2P − FEEDIN) × kWh`; the sum is **gross savings**.
- The **grid-only baseline** is computed on *actual* values: every deficit
  bought at `GRID_PRICE`, every surplus sold at `FEEDIN_PRICE`.
- Trades are committed on *predictions*, so a seller can promise energy it
  does not actually have (e.g. an inverter fault). `undeliveredKwh` is that
  shortfall, and **realized savings** = gross − `undeliveredKwh × (GRID − P2P)`
  (an approximation that ignores seller-side penalties).
- **Perfect-foresight savings** runs the same matcher on actual values: an
  upper bound that shows what forecast accuracy is worth.
- **% of deficit demand met P2P** = confirmed traded kWh ÷ the predicted
  deficit of eligible buyers (also reported against the actual deficit).
- **Prediction MAE** is reported for all hours and excluding *anomaly hours*
  (actual solar = 0 during daylight, or actual consumption = 0). Daylight comes
  from the dataset's reference-only `solar_irradiance` column, which is never a
  model feature.

## 16. Frontend (`frontend/`)

A single-page React + TypeScript dashboard (Vite, Tailwind CSS, recharts,
axios) with two tabs and no router. It is a light, high-contrast theme with
large fonts for projecting in a classroom. Details in
[`frontend/README.md`](frontend/README.md).

**Dashboard tab:** header with the date picker (limited to the replay range),
live connection badges (backend / ML service / chain) and **Run Next Day** /
**Run 1 Hour** buttons; KPI cards for the selected day (predicted vs actual
generation and consumption, energy traded, % of deficit met by P2P, simulated
savings); a 25-household picker with on-chain balances; predicted-vs-actual
solar and consumption charts and a net surplus/deficit bar chart, with data
outage/fault hours highlighted so they are not mistaken for model error; a
next-day forecast panel; a paginated trades table (with a copyable transaction
hash and an "only this household" toggle); and a sortable on-chain balances
table.

**Model Results tab:** the solar and consumption comparison tables (highest R²
highlighted), R² bar charts whose axis starts near the worst model so
differences stay visible, and the overall prediction-error and savings stats.

**Design notes:**
- Each section loads independently, so one failing endpoint shows an inline
  message in that section only. There is no polling; data refreshes after a
  simulation run and when the date or household changes.
- Timestamps are naive local time stored as UTC and are never converted to
  the browser's zone, so hours read 00:00–23:00 on any laptop.
- "Run Next Day" is one ~30 s request with no progress endpoint, so the header
  shows an elapsed-seconds timer rather than a percentage.
- **Mock mode** (`npm run dev:frontend:mock`) serves generated demo data from
  inside the browser, built with the same matching and savings logic and the
  real households' tiers and panel sizes, so the UI can be demoed or developed
  with nothing else running.

Configuration (`frontend/.env`, optional): `VITE_API_URL` (default
`http://localhost:4000`) and `VITE_USE_MOCK` (default `false`).

## 17. What is real and what is simulated

| Real | Simulated or replayed |
|---|---|
| **ML predictions**: live inference from the trained models for every household, every hour | **Smart-meter readings**: "actual" solar and consumption come from the synthetic dataset (§6) |
| **On-chain settlement**: every trade is a real transaction on a (local) EVM chain; balances live in the contract | **Weather**: the dataset's observed weather is replayed as the model input, not a forecast |
| **Persistence**: predictions, trades and the simulation cursor in MongoDB | **Prices**: flat P2P, grid and feed-in tariffs from `.env` (assumed, in Rs.) |
| **Matching and savings maths** on those predictions | **Time**: one simulated hour per tick, from 2026-10-20 00:00 |
| | **Household wallets**: Hardhat's pre-funded default accounts |

Because the weather is the *observed* weather rather than a forecast, the
prediction errors reported here are optimistic compared with a real
deployment, which would feed forecast weather into the same models.
Predictions for dates before 2026-10-20 are in-sample (training data) and are
flagged `inSample: true` by `/forecast`.

## 18. Example run and what it shows

Simulating the first replay day (2026-10-20) with `lightgbm` (pinned at
the time; it is also what `auto` selects from the current tables), as printed by `npm run smoke` against the live stack:

| | |
|---|---|
| Time to simulate 24 slots | about 31 s |
| Trades settled on-chain | 22 (0 failed), 21.84 kWh |
| Deficit demand met P2P | 5.2% of predicted deficit (4.6% of actual) |
| Gross / realized savings | Rs. 109.19 / Rs. 108.71 (undelivered 0.24 kWh) |
| Perfect-foresight upper bound | Rs. 125.04 |
| Grid-only baseline cost | Rs. 1,833.56 (savings ≈ 5.9% of it) |
| Prediction MAE (all hours) | solar 0.083 kWh, consumption 0.182 kWh |

Forecast error cost about 13% of the achievable savings (realized 108.71 vs
the 125.04 bound). The smoke test also confirmed that the on-chain energy and
token balances each sum to 0, that the on-chain trade count equals the
confirmed trades in MongoDB, and that every household's on-chain balances match
the balances derived from MongoDB.

**Why only about 5% of demand is met peer-to-peer.** The market has no
storage, so energy can only move within the same hour. Spot checks of that
day's slots show the structure: at 08:00 some homes already have a surplus
while others still have a deficit (12 trades); at noon 24 of the 25 homes are
in surplus and none is a buyer, so there are no trades; and in the evening and
overnight there are buyers but no sellers (25 of 25 homes are eligible buyers
at 19:00 and 22:00). Trades therefore happen in the morning and
late-afternoon shoulder hours. It is a property of
the no-storage, hourly design, not a bug. (These figures depend on the model;
they will differ slightly with `xgboost` or `catboost`.)

## 19. Testing

| What | Command | Coverage |
|---|---|---|
| Smart contract | `npm run test:contracts` | 10 chai tests: registration (incl. duplicate and zero-address reverts), balance updates on both sides, conservation, integer rounding, owner-only reverts, unregistered/self-trade/zero-value reverts, out-of-range `getTrade` |
| Backend units | `npm run test:backend` | 14 Vitest tests for the matcher (greedy order, conservation, determinism, threshold), savings (including the undelivered-energy penalty and the perfect-foresight bound), MAE with and without anomalies, the integer cost formula, and UTC date parsing |
| End to end | `npm run smoke` | runs one simulated day against the live ML service, chain and MongoDB and asserts the ledger invariants listed in §18 |
| Dashboard | `npm run build:frontend` | strict TypeScript type-check and production build |

The dashboard has no automated UI tests. During development its API layer
was checked against the live backend (every response field the UI reads is
present, and the 404 cases behave as the UI expects), and the whole UI is also
available in mock mode for manual walk-throughs. The crash-recovery path
(reconciling `pending` trades after an interrupted run) is implemented but not
covered by an automated test.

## 20. Trading-layer limitations & future scope

- **No storage, hourly only**: surplus can't be carried to a later hour, which
  caps the P2P share (§18). A battery model is the natural next step.
- **Flat, assumed tariffs**: prices come from `.env` and are not market
  prices. The price function is isolated (`getP2PPriceMilli` in
  `backend/src/services/matching.service.ts`) so dynamic pricing can replace
  it without touching the rest.
- **Optimistic forecasts**: observed weather stands in for a weather forecast
  (§17), and the "actual" meter values are synthetic.
- **Ledger, not money**: credits are signed balances with no escrow, payment
  or real token, and the trusted owner key (Hardhat account #0, publicly known)
  settles everything. The chain is local and in-memory, so it resets whenever
  the node restarts.
- **Approximate realized savings**: the undelivered-energy penalty covers
  sellers only and values the shortfall at `GRID − P2P`.
- **Synchronous API**: a simulated day is one ~30 s HTTP request, with no
  progress endpoint, and `/simulate/run` is capped at 168 hours per call. Only
  one simulation runs at a time.
- **LSTM/ensemble serving is untested here**: `auto` can select `lstm` or
  `ensemble`, and the backend then sends the 24 observed hours before the
  target hour (covered by a unit test), but those models were not trained on
  the machine this was built on, so that path has not run against a live ML
  service. Also note `api/main.py`'s ensemble blends the boosters on the *last
  history row's* features, whereas `train_ensemble.py` scored them on the
  target row's, so the live ensemble may not reproduce its table R²; check it
  before relying on it, or pin `ML_MODEL` to a booster. LSTM/ensemble also
  cannot predict dates without 24 prior hours in the dataset (e.g. 2026-01-01).
- **Dev-only security posture**: no authentication, and CORS allows only the
  Vite dev origin.
- **Future work**: dynamic pricing, battery storage, a real weather-forecast
  input, a public-testnet deployment with per-household wallets, authentication,
  and automated UI tests.


