# Frontend Dashboard (`frontend/`)

A responsive, high-contrast single-page web application built with **React 19**, **TypeScript**, **Vite**, **Tailwind CSS v4**, and **Recharts** to monitor and control the Smart Energy Microgrid Trading platform.

---

## 1. Features & Pages

The application is structured into two main navigation tabs:

### 1. Dashboard Tab (`DashboardPage.tsx`)
- **Service Status Strip:** Real-time health badges for the Backend, ML service (showing the active model, e.g. `ok (ensemble)`), and Ethereum Hardhat blockchain.
- **Daily Summary KPI Row:** Key performance indicators covering solar generation, household consumption, P2P energy traded, realized financial savings, and prediction error (MAE).
- **Interactive Household Picker:** Select any of the 25 households (`#0` to `#24`) to inspect individual performance.
- **Hourly Prediction vs Actual Charts:** Recharts line charts comparing predicted generation/load against actual smart-meter readings.
- **Net Surplus / Deficit Chart:** Hourly bar chart showing surplus solar generation available for peer-to-peer trading.
- **Day-Ahead Forecast Panel:** Solar and consumption projection for the following 24 hours.
- **Trades Card (`TradesTable.tsx`):** Paginated tabular log of executed peer-to-peer trades with slot timestamps, seller/buyer IDs, energy volume (kWh), agreed price, total value, transaction status, and on-chain transaction hashes.
- **Interactive P2P Trading Network Graph (`TradesNetworkGraph.tsx`):**
  - Graph visualization placed directly below the Trades Card.
  - **House Symbol Nodes:** Microgrid households are represented by clean SVG house symbols.
  - **Trade Flow Color Coding:**
    - 🟢 **Green Arrows:** Incoming energy purchased by the household.
    - 🔴 **Red Arrows:** Outgoing energy sold by the household.
  - **Zero-Jitter Hit Area:** Fixed transparent hit-testing circles ensure smooth, flicker-free mouse interaction.
  - **View Modes:**
    - **Network Ring:** 25-house circular perimeter with curved directional energy chords.
    - **Hub View:** Centers the selected household with its active trading partners arranged radially.
  - **Time Scrubber & Animation:** Hour buttons (`00:00` to `23:00`), "All 24 Hours", and a **`▶ Play Day`** button to automatically animate hourly energy transfers.
  - **Click-to-Select:** Clicking any house node switches the selected household across the whole dashboard.
- **On-Chain Balances Table (`BalancesTable.tsx`):** Displays real-time double-entry energy (kWh) and token (INR) balances retrieved directly from the Solidity smart contract.

### 2. Model Results Tab (`ModelResultsPage.tsx`)
- Comprehensive benchmark comparison across all evaluated ML models (Baselines, LightGBM, CatBoost, LSTM, Stacking Ensemble, and Prophet).
- Displays RMSE, MAE, and $R^2$ scores, alongside visual bar charts highlighting model accuracy improvements.

---

## 2. Tech Stack

- **Framework:** React 19 + TypeScript (strict mode)
- **Bundler & Dev Server:** Vite 8 (pinned to port `5173`)
- **Styling:** Tailwind CSS v4 (clean, high-contrast light theme)
- **Charts:** Recharts
- **HTTP Client:** Axios (with custom mock adapter support)

---

## 3. Getting Started

### Installation
From the repository root:
```bash
npm --prefix frontend install
```
Or directly inside `frontend/`:
```bash
npm install
```

### Development Server
```bash
npm run dev
# Starts local dev server at http://localhost:5173
```

> **Strict Port 5173:** The backend CORS policy permits requests from `http://localhost:5173`. If port 5173 is occupied, Vite will report an error rather than silently hopping to port 5174.

### Production Build
Type-checks TypeScript and compiles production assets to `dist/`:
```bash
npm run build
```

---

## 4. Environment Variables (`.env`)

Copy `frontend/.env.example` to `frontend/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | `http://localhost:4000` | Backend API base URL |
| `VITE_USE_MOCK` | `false` | When `true`, runs the dashboard entirely in mock mode |

---

## 5. Mock Mode (Stand-alone Demo)

You can launch and explore the full frontend without running MongoDB, the Hardhat node, or the Python ML service:

```bash
npm run dev:mock       # from frontend/ or repo root
```

In mock mode:
- A custom in-browser Axios adapter (`src/api/mock/`) simulates backend responses.
- Generates realistic 25-household data, historical simulated days, predictions, and on-chain mock trades.
- Clicking **Run Next Day** advances the simulated date with simulated latency.
- An amber indicator badge in the header clearly indicates mock mode.

---

## 6. Directory Structure

```
frontend/src/
├── api/
│   ├── client.ts              # Axios instance configuration
│   ├── endpoints.ts           # Typed API endpoint functions
│   └── mock/                  # Standalone mock adapter and data generator
├── components/
│   ├── ActiveModelsCard.tsx   # Model selection cards
│   ├── AsyncState.tsx         # Loading, error, and empty-state wrapper
│   ├── Badge.tsx              # Colored status badges
│   ├── BalancesTable.tsx      # On-chain smart contract balance table
│   ├── ForecastPanel.tsx      # Day-ahead forecast charts
│   ├── Header.tsx             # App header, simulation controls, status badges
│   ├── HealthBadges.tsx       # Live status indicators (Backend, ML, Chain)
│   ├── HouseholdPicker.tsx    # Household selector buttons
│   ├── KpiRow.tsx             # Daily summary KPI cards
│   ├── NetChart.tsx           # Net surplus / deficit bar chart
│   ├── PredActualChart.tsx    # Predicted vs actual generation/load charts
│   ├── Section.tsx            # Standard card section container
│   ├── TradesNetworkGraph.tsx # P2P energy trading network graph visual
│   └── TradesTable.tsx        # Paginated trades log table
├── hooks/
│   ├── EpochContext.tsx       # Global simulation refresh context
│   ├── resources.ts           # Cached data hooks (useSummary, useAllTradesForDate, ...)
│   ├── useResource.ts         # Generic cached resource fetcher
│   └── useSimulation.ts       # Simulation action hook (runDay, reset)
├── pages/
│   ├── DashboardPage.tsx      # Main microgrid trading dashboard
│   └── ModelResultsPage.tsx   # Model evaluation and comparison page
├── types/
│   └── api.ts                 # TypeScript interfaces matching backend response schemas
└── utils/
    ├── chartData.ts           # Time-series transformations for Recharts
    ├── date.ts                # UTC date manipulation helpers
    └── format.ts              # Currency (Rs.), energy (kWh), and percentage formatters
```
