# Backend Service (`backend/`)

Express/TypeScript service that orchestrates the smart energy microgrid simulation: it feeds hourly meter data to the ML service, computes peer-to-peer surplus and deficits, runs a greedy double-auction matching engine, settles trades on the local Ethereum smart contract, and persists state in MongoDB.

---

## 1. Architecture

```
                 +-------------------------+
 energy_data.csv |  dataSource.service     |  weather, panel, tier, actuals (replayed)
 (ml-service) -->|  25 households / hour   |
                 +-----------+-------------+
                             | slot rows
                             v
   POST /predict/*   +-------------------+        +-----------------------+
 ML service :8000 <--| mlClient.service  |        |  MongoDB (Mongoose)   |
 (FastAPI, live      | (with Fallback    |        |  - Prediction         |
  inference)         |  Chain mechanism) |        |  - Trade              |
                     +---------+---------+        |  - SimulationState    |
                               | predictions      |    (cursor survives   |
                               v                  |     restarts)         |
                     +-------------------+  save  +-----------------------+
                     | simulation.service|------->
                     |  runSlot / runDay |
                     +----+---------+----+
              surplus/    |         | trades, one at a time
              deficit     v         v
          +--------------------+  +------------------------+   ethers.js    +------------------+
          | matching.service   |  | blockchain.service     |--------------->| Hardhat node     |
          | greedy, flat price |  | owner signer, receipts |  JSON-RPC      | EnergyTrading.sol|
          +--------------------+  +------------------------+  :8545         | (on-chain ledger)|
                                                                             +------------------+

   HTTP API (Express :4000): /simulate/*, /trades, /households, /predictions, /forecast,
   /daily-summary, /stats, /results/models, /models/active, /health
```

---

## 2. Core Modules & Services

### 1. `dataSource.service.ts`
- Ingests `ml-service/data/energy_data.csv` (1 full year of 25 households at 1-hour resolution).
- Replays historical test records (starting Oct 20, 2026 00:00).
- Provides 24-hour sliding window history for sequence models (`lstm`, `ensemble`).

### 2. `mlClient.service.ts` & Fallback Engine
- Sends prediction requests to FastAPI on `:8000`.
- **Model Fallback Chain (`tryWithFallback`):**
  If the preferred model fails (e.g. `ensemble` encounters missing history on Day 1 or an internal exception), the client automatically downgrades to the next best model in sequence:
  $$\text{ensemble} \longrightarrow \text{lstm} \longrightarrow \text{lightgbm} \longrightarrow \text{xgboost}$$
- **Concurrency Control:** Limits sequence model requests (`ML_SEQUENCE_CONCURRENCY = 1`) to avoid thread-pool saturation in TensorFlow/Keras, while batching tabular models (`BATCH_SIZE = 25`).

### 3. `matching.service.ts`
- Computes hourly net position for each household:
  $$\text{net} = \text{predictedSolar} - \text{predictedConsumption}$$
- Identifies **sellers** ($\text{net} > \text{MIN\_KWH}$) and **buyers** ($\text{net} < -\text{MIN\_KWH}$).
- **Greedy Double Auction:** Repeatedly matches the largest remaining surplus seller with the largest remaining deficit buyer at the fixed P2P tariff (`P2P_PRICE_PER_KWH`, e.g. Rs. 5.50/kWh).

### 4. `blockchain.service.ts`
- Uses `ethers.js` (v6) to communicate with `EnergyTrading.sol` on Hardhat (`:8545`).
- Executes trades atomically: validates that both seller and buyer are registered, updates on-chain milli-kWh energy balances and token balances, and retrieves transaction hashes.

### 5. `health.service.ts`
- Probes MongoDB, Hardhat blockchain connection, ML service health, and verifies that on-chain trade counts match MongoDB trade records (`chainDbSync`).
- Reports the active ML model serving predictions (`activeModel`).

---

## 3. Environment Variables (`.env`)

Copy `backend/.env.example` to `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `4000` | Port for Express server |
| `MONGO_URI` | `mongodb://127.0.0.1:27017/energy_trading` | MongoDB connection string (local or Atlas) |
| `ML_SERVICE_URL` | `http://127.0.0.1:8000` | FastAPI ML service endpoint |
| `HARDHAT_RPC_URL` | `http://127.0.0.1:8545` | Hardhat JSON-RPC node |
| `OWNER_PRIVATE_KEY` | `0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80` | Account #0 deployer key |
| `ML_MODEL` | *(auto highest R²)* | Optional global model override: `ensemble`, `lstm`, `lightgbm`, `xgboost`, `catboost` |
| `ML_MODEL_SOLAR` | `""` | Target-specific override for solar forecasting |
| `ML_MODEL_CONSUMPTION` | `""` | Target-specific override for consumption forecasting |
| `P2P_PRICE_PER_KWH` | `5.50` | Tariff for peer-to-peer microgrid trades (INR/kWh) |
| `GRID_PRICE_PER_KWH` | `8.00` | Retail grid import tariff (INR/kWh) |
| `FEEDIN_PRICE_PER_KWH` | `3.00` | Grid feed-in tariff (INR/kWh) |
| `MIN_TRADE_KWH` | `0.05` | Minimum threshold for trading eligibility |

---

## 4. API Reference

### Simulation Control
- `POST /simulate/tick`: Simulates a single hourly slot.
- `POST /simulate/day`: Simulates up to 24 slots (completes the current calendar day) and returns summary.
- `POST /simulate/run?hours=N`: Simulates $N$ consecutive slots ($1 \le N \le 168$).
- `POST /simulate/reset`: Resets MongoDB simulation state (Predictions, Trades, and cursor).

### Data & Query Routes
- `GET /daily-summary?date=YYYY-MM-DD`: Totals, savings, MAE, and % deficit met P2P.
- `GET /households`: 25 registered households, solar capacity, tiers, and live on-chain balances.
- `GET /trades?page=1&limit=50&date=YYYY-MM-DD&status=confirmed`: Paginated trade records.
- `GET /trades/household/:id`: All trades involving a specific household.
- `GET /predictions/:householdId?date=YYYY-MM-DD`: Hourly predicted vs actual solar & consumption.
- `GET /forecast/:householdId?date=YYYY-MM-DD`: Day-ahead forecast series.
- `GET /stats`: Cumulative microgrid statistics across all simulated days.
- `GET /models/active`: Returns active model and fallback chain order per target.
- `GET /results/models`: Comparative model performance tables (RMSE, MAE, R²).
- `GET /health`: Comprehensive health check of all microgrid components.

---

## 5. Development & Testing

### Running Tests
```bash
npm run test:backend       # from repo root
# or:
npm test                   # inside backend/
```

### End-to-End Smoke Test
Verifies end-to-end integration across ML service, blockchain settlement, double-entry balance conservation, and database writes:
```bash
npm run smoke
```
