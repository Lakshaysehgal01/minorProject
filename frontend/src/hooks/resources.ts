import {
  getActiveModels,
  getDailySummary,
  getForecast,
  getHealth,
  getHouseholdTrades,
  getHouseholds,
  getModelResults,
  getPredictions,
  getStats,
  getTrades,
} from "../api/endpoints";
import type { Trade } from "../types/api";
import { useResource } from "./useResource";


export const useHealth = () => useResource("health", (s) => getHealth(s));


export const useStats = () => useResource("stats", (s) => getStats(s));


export const useHouseholds = () => useResource("households", (s) => getHouseholds(s));


export const useSummary = (date: string | null) =>
  useResource(date ? `summary:${date}` : null, (s) => getDailySummary(date as string, s));


export const usePredictions = (householdId: number | null, date: string | null) =>
  useResource(householdId !== null && date ? `pred:${householdId}:${date}` : null, (s) =>
    getPredictions(householdId as number, date as string, s)
  );


export const useForecast = (householdId: number | null, date: string | null) =>
  useResource(householdId !== null && date ? `forecast:${householdId}:${date}` : null, (s) =>
    getForecast(householdId as number, date as string, s)
  );


export const useTrades = (date: string | null, page: number, limit: number, enabled = true) =>
  useResource(date && enabled ? `trades:${date}:${page}:${limit}` : null, (s) =>
    getTrades({ page, limit, date: date as string }, s)
  );


// /trades/household/:id has no date filter, so fetch all of the household's trades (pages of 500)
// and let the caller filter by date client-side.
export const useHouseholdTrades = (householdId: number | null, enabled = true) =>
  useResource<Trade[]>(householdId !== null && enabled ? `hhtrades:${householdId}` : null, async (signal) => {
    const all: Trade[] = [];
    for (let page = 1; ; page++) {
      const res = await getHouseholdTrades(householdId as number, { page, limit: 500 }, signal);
      all.push(...res.trades);
      if (res.trades.length === 0 || all.length >= res.total) break;
    }
    return all;
  });


export const useActiveModels = () => useResource("active-models", (s) => getActiveModels(s));


export const useModelResults = () => useResource("model-results", (s) => getModelResults(s));



