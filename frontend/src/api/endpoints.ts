import { api, SIMULATION_TIMEOUT_MS } from "./client";
import type {
  ActiveModels,
  DailySummary,
  ForecastResponse,
  HealthResponse,
  Household,
  ModelResults,
  PredictionsResponse,
  RunDayResponse,
  SlotSummary,
  Stats,
  TradesPage,
} from "../types/api";


export const runDay = () =>
  api.post<RunDayResponse>("/simulate/day", null, { timeout: SIMULATION_TIMEOUT_MS }).then((r) => r.data);


export const runTick = () =>
  api.post<SlotSummary>("/simulate/tick", null, { timeout: SIMULATION_TIMEOUT_MS }).then((r) => r.data);


export const getDailySummary = (date: string, signal?: AbortSignal) =>
  api.get<DailySummary>("/daily-summary", { params: { date }, signal }).then((r) => r.data);


export const getHouseholds = (signal?: AbortSignal) =>
  api.get<Household[]>("/households", { signal }).then((r) => r.data);


export const getTrades = (params: { page: number; limit: number; date?: string }, signal?: AbortSignal) =>
  api.get<TradesPage>("/trades", { params, signal }).then((r) => r.data);


export const getHouseholdTrades = (id: number, params: { page: number; limit: number }, signal?: AbortSignal) =>
  api.get<TradesPage>(`/trades/household/${id}`, { params, signal }).then((r) => r.data);


export const getPredictions = (householdId: number, date: string, signal?: AbortSignal) =>
  api.get<PredictionsResponse>(`/predictions/${householdId}`, { params: { date }, signal }).then((r) => r.data);


export const getForecast = (householdId: number, date: string, signal?: AbortSignal) =>
  api.get<ForecastResponse>(`/forecast/${householdId}`, { params: { date }, signal }).then((r) => r.data);


export const getStats = (signal?: AbortSignal) => api.get<Stats>("/stats", { signal }).then((r) => r.data);


export const getModelResults = (signal?: AbortSignal) =>
  api.get<ModelResults>("/results/models", { signal }).then((r) => r.data);


export const getActiveModels = (signal?: AbortSignal) =>
  api.get<ActiveModels>("/models/active", { signal }).then((r) => r.data);


// The backend answers 503 with a full body when a component is down: that is data, not a failure.
export const getHealth = (signal?: AbortSignal) =>
  api
    .get<HealthResponse>("/health", { signal, validateStatus: (s) => s === 200 || s === 503 })
    .then((r) => r.data);



