# Blockchain + Backend layer


Express/TypeScript backend and a Hardhat smart contract that replay the Oct 20 – Dec 31 2026 test period of the ML dataset hour by hour: the ML service predicts each household's solar and consumption, a greedy matcher pairs surplus with deficit, and trades settle on a **local** Hardhat chain.


The existing ML pipeline README is at the repo root. `/ml-service` is not modified by this layer.


## Architecture


```
                 +-------------------------+
 energy_data.csv |  dataSource.service     |  weather, panel, tier, actuals (replayed)
 (ml-service) -->|  25 households / hour   |
                 +-----------+-------------+
                             | slot rows
                             v
   POST /predict/*   +-------------------+        +-----------------------+
 ML service :8000 <--| mlClient.service  |        |  MongoDB (Mongoose)   |
 (FastAPI, real      +---------+---------+        |  Prediction           |
  model inference)             | predictions      |  Trade                |
                               v                  |  SimulationState      |
                     +-------------------+  save  |  (cursor survives     |
                     | simulation.service|------->|   restarts)           |
                     |  runSlot/runDay   |        +-----------------------+
                     +----+---------+----+
              surplus/    |         | trades, one at a time
              deficit     v         v
          +--------------------+  +------------------------+   ethers.js    +------------------+
          | matching.service   |  | blockchain.service     |--------------->| Hardhat node     |
          | greedy, flat price |  | owner signer, receipts |  JSON-RPC      | EnergyTrading.sol|
          +--------------------+  +------------------------+  :8545         | (on-chain ledger)|
                                                                             +------------------+
   HTTP API (Express :4000): /simulate/*, /trades, /households, /predictions, /forecast,
   /daily-summary, /stats, /results/models, /health
```


Trading runs **hourly**: solar surplus at noon cannot offset an evening deficit without storage. "Daily" is only an aggregation of 24 hourly ticks.


## Prerequisites


- Node 20+ (tested on Node 24) and npm
- A MongoDB you can reach (local install or Atlas). Put its URI in `backend/.env`
- The ML service artifacts (all gitignored, generated from `ml-service/` with its venv activated):
  ```
  python data/generate_data.py
  python train.py
  python models/train_lightgbm.py
  python models/train_catboost.py
  ```
  `generate_data.py` downloads one year of Delhi weather from Open-Meteo once. Behind a TLS-inspecting proxy this can fail with `SSLCertVerificationError`; see the note in `ml-service/data/fetch_weather.py` (fetch the weather cache file from another network or tool and place it at `ml-service/data/weather_cache/delhi_weather_2024.csv`).


## Run order


All commands from the **repo root** (they delegate to `/contracts` and `/backend` and work on Windows, macOS and Linux).


```
npm run install:all                     # once
copy backend\.env.example backend\.env  # (cp on macOS/Linux) then set MONGO_URI


# terminal 1: local blockchain
npm run chain


# terminal 2: deploy + register 25 households (writes backend/src/contracts/EnergyTrading.json)
npm run deploy


# terminal 3: ML service (from ml-service/, venv activated)
uvicorn api.main:app --port 8000


# terminal 4: backend (Mongo must be reachable)
npm run dev


# then simulate one day
curl -X POST http://localhost:4000/simulate/day
# or run the end-to-end check, which also verifies ledger invariants
npm run smoke
```


Other scripts: `npm run test:contracts`, `npm run test:backend`, `npm run build`.


> The Hardhat node is **in-memory**. If you restart `npm run chain`, run `npm run deploy` again and then `POST /simulate/reset` (the backend refuses to run while Mongo is ahead of the chain, and `/health` reports it).


## API


| Method | Route | Notes |
|---|---|---|
| POST | `/simulate/tick` | one hourly slot |
| POST | `/simulate/day` | runs until the next 00:00 (24 slots from midnight), returns the day summary |
| POST | `/simulate/run?hours=N` | N slots (1-168) |
| POST | `/simulate/reset` | clears Predictions, Trades and the cursor |
| GET | `/daily-summary?date=YYYY-MM-DD` | totals, % deficit met P2P, savings, MAE |
| GET | `/households` | id, tier, panel kW, on-chain energy + token balance |
| GET | `/trades?page&limit&date&status` | newest first |
| GET | `/trades/household/:id` | |
| GET | `/predictions/:householdId?date=` | hourly predicted vs actual |
| GET | `/forecast/:householdId?date=` | predictions only, no actuals; see caveat below |
| GET | `/stats` | overall totals, savings, MAE |
| GET | `/results/models` | model comparison tables from `ML_RESULTS_DIR` |
| GET | `/models/active` | model in use per target (auto-selected by highest R², or pinned) |
| GET | `/health` | Mongo, ML (real probe prediction), chain, chain/DB sync |


Status codes: 400 validation, 404 no data, 409 busy / complete / chain-DB mismatch, 502 bad ML response, 503 a dependency is down.


## Definitions


- **Units on-chain**: energy in milli-kWh, prices in milli-credits per kWh. `cost = kWhMilli * pricePerKwhMilli / 1000` (integer division). Token balances are signed ledger entries, so a net buyer can be negative.
- **Matching**: `surplus = predictedSolar - predictedConsumption`. Sellers have surplus > `MIN_KWH`, buyers a deficit > `MIN_KWH` (eligibility only; an eligible household trades its full surplus/deficit). The largest remaining seller is repeatedly matched to the largest remaining buyer (ties: lower household id), at the flat `P2P_PRICE_PER_KWH`.
- **Savings** (`GRID_PRICE > P2P_PRICE_PER_KWH > FEEDIN_PRICE`): per confirmed trade the buyer saves `(GRID - P2P) * kWh` and the seller gains `(P2P - FEEDIN) * kWh`; their sum is **gross savings**.
- **Grid-only baseline**: on *actual* values, every deficit bought at `GRID_PRICE` and every surplus sold at `FEEDIN_PRICE`.
- **Realized savings**: trades are committed on predictions. `undeliveredKwh` is energy a seller sold beyond its actual surplus (e.g. an inverter fault); `realizedSavings = gross - undelivered * (GRID - P2P)`. This is an approximation (it ignores seller-side penalties).
- **Perfect-foresight savings**: the same matcher run on actuals, an upper bound that shows what forecast accuracy is worth.
- **% deficit met P2P**: confirmed traded kWh divided by the predicted deficit of eligible buyers (the demand the market saw); also reported against actual deficit.
- **Prediction MAE**: all hours, and excluding anomaly hours (actual solar == 0 in daylight, or actual consumption == 0). Daylight is taken from the dataset's reference-only `solar_irradiance` column (never a model feature).
- **Predictions are clamped to >= 0.**


### What the market actually looks like


With no storage, trades only occur in the shoulder hours (roughly morning and late afternoon). At midday nearly every household with panels is in surplus (no buyers) and overnight every household is a buyer (no sellers), so the share of total deficit met P2P is modest. That is a property of the no-storage hourly design, not a bug.


## What is simulated vs real


| Simulated / replayed | Real |
|---|---|
| Smart-meter readings: actual solar and consumption come from the synthetic dataset | ML predictions: live inference from the trained models in `ml-service` |
| Weather: the dataset's weather columns for each hour are replayed as the model input | Settlement: every trade is a real transaction on a (local) EVM chain, with balances held in the contract |
| Prices: flat P2P, grid and feed-in prices from `.env` | Persistence: predictions, trades and the simulation cursor in MongoDB |
| Time: one hourly tick per request, starting 2026-10-20 00:00 | |
| Household wallets: Hardhat's default funded accounts | |


**Weather caveat.** The weather fed to the models is the *observed* weather for that hour, not a forecast. A real deployment would need a weather forecast API, so the prediction errors reported here are optimistic. `/forecast` documents this and flags `inSample: true` for dates before 2026-10-20 (those dates were part of the models' training data).


## Notes


- `lstm` and `ensemble` models need a 24-row history payload and are not used. Choose `xgboost`, `lightgbm` or `catboost` with `ML_MODEL` (or per target with `ML_MODEL_SOLAR` / `ML_MODEL_CONSUMPTION`).
- A slot is processed idempotently: predictions are upserted, trades are written as `pending` before sending and reconciled against on-chain `TradeExecuted` events if a run was interrupted, and the cursor advances last.
- Only one simulation request runs at a time; a concurrent one gets `409`.
- `OWNER_PRIVATE_KEY` defaults to Hardhat account #0, whose key is public. Never use it on a real network.
- `EnergyTrading.sol` takes an extra `slotTimestamp` argument so the chain records simulated time, not wall-clock time.
