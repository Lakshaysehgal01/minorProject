# Dashboard (frontend)


React + TypeScript + Vite single-page dashboard for the Smart Energy Microgrid Trading demo. Two tabs, **Dashboard** and **Model Results**, light high-contrast theme sized for a projector. It only reads from the backend's HTTP API; nothing in `/backend` or `/ml-service` is changed by it.


Stack: React 19, TypeScript (strict), Vite, Tailwind CSS v4, recharts, axios.


## Install and run


```
cd frontend
npm install
copy .env.example .env      # (cp on macOS/Linux)
npm run dev                 # http://localhost:5173
```


`npm run build` type-checks and produces `dist/`.


The dev server is pinned to **port 5173** (`strictPort`): the backend only allows CORS from `http://localhost:5173`, so if the port is busy Vite fails instead of silently moving to 5174.


### Environment


| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_URL` | `http://localhost:4000` | Backend base URL. The backend listens on **4000** by default (not 5000); change this if you changed `PORT` in `backend/.env`. |
| `VITE_USE_MOCK` | `false` | `true` serves generated demo data inside the browser. |


## Demo order (real data)


1. Hardhat node: `npm run chain` (repo root)
2. Deploy the contract: `npm run deploy` (repo root)
3. ML service: `uvicorn api.main:app --port 8000` from `ml-service/` with its venv active
4. MongoDB reachable (local or Atlas; `MONGO_URI` in `backend/.env`)
5. Backend: `npm run dev` (repo root)
6. Frontend: `npm run dev:frontend` (repo root) or `npm run dev` in `frontend/`
7. Open http://localhost:5173 and click **Run Next Day**


For a clean demo start, call `POST /simulate/reset` on the backend, and restart the Hardhat node and redeploy so the on-chain ledger is empty too (the backend refuses to run while Mongo is ahead of the chain).


Check the green/red connection badges in the header first. They come from `GET /health`, which does a real probe prediction against the ML service.


## Mock mode


```
npm run dev:mock            # cross-platform; sets VITE_USE_MOCK=true
```


Requests are answered in the browser by a custom axios adapter (`src/api/mock`), so the UI code is identical in both modes. Data is deterministic and built the same way as the real system (greedy matching, savings, anomaly flags, 25 households with the real tiers and panel sizes). It starts with two days already simulated; **Run Next Day** takes about 3 s and advances the mock cursor. An amber banner marks mock mode.


Extra switches for testing empty/error states (set in `.env.local`):
- `VITE_MOCK_HEALTH=degraded`: the health call returns 503 with the ML service down
- `VITE_MOCK_NO_RESULTS=true`: `/results/models` returns 404


## Things worth knowing


- **Time:** the backend stores naive local time as UTC. The UI never converts to the browser's time zone, so hours read 00:00 to 23:00 on any laptop.
- **Run Next Day** completes the current simulated day (if you ran single hours first, it finishes that day). A day is one ~30 s request; there is no progress endpoint, so the header shows an elapsed timer instead of a percentage.
- **Savings** are labelled "simulated savings (assumed tariffs)": realized savings after forecast-error penalties, with gross savings and the perfect-foresight upper bound underneath.
- **% deficit met by P2P is low** because there is no storage: trades only happen in shoulder hours when some homes have a surplus and others a deficit.
- **Forecast** uses the dataset's weather columns as input. A real deployment would need a weather forecast API.
- **Household trades on a date:** `GET /trades/household/:id` has no date filter, so the toggle fetches the household's trades and filters by the selected date in the browser.
- Each section loads independently: one failing endpoint shows an inline message in that section only. Responses are cached per date/household and cleared after every simulation run; there is no polling.
- **Model Results** reads CSVs produced by `python results/model_comparison.py` in `ml-service`. If they are missing the tab shows "Run the ML training and model_comparison.py to generate results."


## Structure


```
src/
  api/         axios instance, typed endpoint functions, mock adapter + data
  components/  KPI cards, charts, tables, header, async-state wrapper
  hooks/       one hook per resource (useSummary, useTrades, ...), useSimulation
  pages/       DashboardPage, ModelResultsPage
  types/       API types, taken from the backend's response shapes
  utils/       formatting, UTC date helpers, chart data transforms, cache
```



