import axios, { AxiosError } from "axios";
import { config } from "../config/env";
import { HouseholdRow, getHistory } from "./dataSource.service";
import { getSelection, isSequenceModel, getFallbackOrder, type ModelName, type Target } from "./modelSelection.service";
import { HttpError, httpError } from "../utils/errors";


const BATCH_SIZE = 25;


const http = axios.create({ baseURL: config.ML_SERVICE_URL, timeout: config.ML_TIMEOUT_MS });


const timeoutFor = (model: string): number => (isSequenceModel(model) ? config.ML_SEQUENCE_TIMEOUT_MS : config.ML_TIMEOUT_MS);


export function solarPayload(r: HouseholdRow) {
  return {
    hour: r.hour,
    day_of_week: r.dayOfWeek,
    day_of_year: r.dayOfYear,
    month: r.month,
    household_id: r.householdId,
    temperature: r.temperature,
    cloud_cover: r.cloudCover,
    humidity: r.humidity,
    wind_speed: r.windSpeed,
    panel_capacity_kw: r.panelCapacityKw,
  };
}


export function consumptionPayload(r: HouseholdRow) {
  return {
    hour: r.hour,
    day_of_week: r.dayOfWeek,
    household_id: r.householdId,
    temperature: r.temperature,
    humidity: r.humidity,
    is_holiday: r.isHoliday,
    consumption_profile_encoded: r.tierEncoded,
  };
}


function toMlError(err: unknown): HttpError {
  const ax = err as AxiosError<{ detail?: unknown }>;
  if (!ax.isAxiosError) return httpError(502, `ML service error: ${(err as Error).message}`);
  if (!ax.response) {
    const secs = Math.round((ax.config?.timeout ?? config.ML_TIMEOUT_MS) / 1000);
    const reason = ax.code === "ECONNABORTED" || ax.code === "ETIMEDOUT" ? `timed out after ${secs}s` : "unreachable";
    return httpError(503, `ML service ${reason} at ${config.ML_SERVICE_URL}. Is uvicorn running? (${ax.code ?? ax.message})`);
  }
  const detail = typeof ax.response.data?.detail === "string" ? ax.response.data.detail : JSON.stringify(ax.response.data);
  if (ax.response.status === 503) return httpError(503, `ML model not available: ${detail}`);
  return httpError(502, `ML service returned ${ax.response.status}: ${detail}`);
}


function clamp(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw httpError(502, `ML service returned a non-numeric ${label}: ${JSON.stringify(value)}`);
  }
  return Math.max(0, value);
}


// lstm/ensemble predict hour h from the 24 observed hours BEFORE it (rows h-24..h-1, each with its
// actual target), so no future value leaks in. Single-row models take just the hour's own features.
function getHistoryIfAvailable(row: HouseholdRow): HouseholdRow[] | null {
  return getHistory(row.householdId, row.timestampMs);
}


export function solarBody(row: HouseholdRow, model: string, history: HouseholdRow[] | null) {
  if (!isSequenceModel(model)) return solarPayload(row);
  if (!history) return null; // signal: cannot build sequence payload
  return { rows: history.map((r) => ({ ...solarPayload(r), solar_generation_kwh: r.actualSolarKwh })) };
}


export function consumptionBody(row: HouseholdRow, model: string, history: HouseholdRow[] | null) {
  if (!isSequenceModel(model)) return consumptionPayload(row);
  if (!history) return null; // signal: cannot build sequence payload
  return {
    rows: history.map((r) => ({
      ...consumptionPayload(r),
      // day_of_year is required by the LSTM's add_cyclical_features() even though
      // the consumption tabular models don't use it — it computes doy_sin/doy_cos
      // unconditionally. Not sent for single-row (tabular) requests.
      day_of_year: r.dayOfYear,
      consumption_kwh: r.actualConsumptionKwh,
    })),
  };
}


/**
 * Try each model in the fallback order for a target, returning the first successful prediction.
 * Reasons for skipping each model are collected and logged at the end.
 */
async function tryWithFallback(
  target: Target,
  row: HouseholdRow,
  buildBody: (model: ModelName, history: HouseholdRow[] | null) => object | null,
  endpoint: string,
  resultKey: string
): Promise<{ value: number; modelUsed: ModelName }> {
  const order = getFallbackOrder(target);
  const history = getHistoryIfAvailable(row);
  const skipped: string[] = [];

  for (const model of order) {
    // If this is a sequence model but history is unavailable, skip it immediately.
    if (isSequenceModel(model) && !history) {
      skipped.push(`${model}: skipped (no 24h history before ${new Date(row.timestampMs).toISOString()})`);
      continue;
    }

    const body = buildBody(model, history);
    if (body === null) {
      skipped.push(`${model}: skipped (could not build payload)`);
      continue;
    }

    try {
      const res = await http.post(endpoint, body, { params: { model }, timeout: timeoutFor(model) });
      if (skipped.length > 0) {
        console.warn(`[mlClient] fallback for household ${row.householdId} @ ${new Date(row.timestampMs).toISOString()}: ${skipped.join("; ")} → succeeded with ${model}`);
      }
      return { value: clamp(res.data?.[resultKey], `${target} prediction`), modelUsed: model };
    } catch (err) {
      const httpErr = (err as HttpError).status ? (err as HttpError) : toMlError(err);
      // If the ML service host is completely unreachable (ECONNREFUSED / down), propagate immediately.
      const ax = err as AxiosError;
      if (ax?.isAxiosError && !ax.response && (ax.code === "ECONNREFUSED" || ax.code === "ENOTFOUND")) {
        throw httpErr;
      }
      // Otherwise (500 internal server error in model, 502, 503 unavailable, 422 invalid payload/history, or timeout),
      // record reason and fall back to the next best model in sequence
      skipped.push(`${model}: ${httpErr.message}`);
    }
  }

  // All models exhausted
  const allReasons = skipped.join(" | ");
  throw httpError(503, `All models failed for ${target} prediction (household ${row.householdId}): ${allReasons}`);
}


export async function getSolarPrediction(row: HouseholdRow, model: string = getSelection("solar").model): Promise<number> {
  // Single explicit model (not using fallback chain) — keep the original simple path for direct calls.
  if (!getFallbackOrder("solar").includes(model as ModelName)) {
    try {
      const history = getHistoryIfAvailable(row);
      const body = solarBody(row, model, history);
      if (body === null) throw httpError(422, `Model "${model}" needs 24 hours of history before ${new Date(row.timestampMs).toISOString()}, but the dataset does not go back that far.`);
      const res = await http.post("/predict/solar", body, { params: { model }, timeout: timeoutFor(model) });
      return clamp(res.data?.predicted_solar_generation_kwh, "solar prediction");
    } catch (e) {
      throw (e as HttpError).status ? e : toMlError(e);
    }
  }
  const { value } = await tryWithFallback(
    "solar",
    row,
    (m, h) => solarBody(row, m, h),
    "/predict/solar",
    "predicted_solar_generation_kwh"
  );
  return value;
}


export async function getConsumptionPrediction(row: HouseholdRow, model: string = getSelection("consumption").model): Promise<number> {
  if (!getFallbackOrder("consumption").includes(model as ModelName)) {
    try {
      const history = getHistoryIfAvailable(row);
      const body = consumptionBody(row, model, history);
      if (body === null) throw httpError(422, `Model "${model}" needs 24 hours of history before ${new Date(row.timestampMs).toISOString()}, but the dataset does not go back that far.`);
      const res = await http.post("/predict/consumption", body, { params: { model }, timeout: timeoutFor(model) });
      return clamp(res.data?.predicted_consumption_kwh, "consumption prediction");
    } catch (e) {
      throw (e as HttpError).status ? e : toMlError(e);
    }
  }
  const { value } = await tryWithFallback(
    "consumption",
    row,
    (m, h) => consumptionBody(row, m, h),
    "/predict/consumption",
    "predicted_consumption_kwh"
  );
  return value;
}


export interface RowPrediction {
  householdId: number;
  predictedSolarKwh: number;
  predictedConsumptionKwh: number;
}


export function modelLabel(): string {
  const solar = getSelection("solar").model;
  const consumption = getSelection("consumption").model;
  return solar === consumption ? solar : `solar=${solar};consumption=${consumption}`;
}


// Runs fn over items with at most `limit` in flight, preserving order.
async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i]);
    }
  });
  await Promise.all(workers);
  return results;
}


// lstm/ensemble run a shared Keras model inside the ML service's thread pool, which is not safe to
// hit with many simultaneous requests (it can crash the process: ECONNRESET). The single-row
// boosters are fine in parallel.
function usesSequenceModel(): boolean {
  return isSequenceModel(getSelection("solar").model) || isSequenceModel(getSelection("consumption").model);
}


// Predict a set of households (the ML API is single-row, so one request per household and target).
export async function predictRows(rows: HouseholdRow[]): Promise<RowPrediction[]> {
  // When the primary model is a sequence model, run sequentially to avoid thread-pool races;
  // but the fallback mechanism may silently downgrade to a tabular model for early rows.
  const sequence = usesSequenceModel();
  const limit = sequence ? config.ML_SEQUENCE_CONCURRENCY : BATCH_SIZE;
  return mapLimit(rows, limit, async (r) => {
    if (sequence) {
      const predictedSolarKwh = await getSolarPrediction(r);
      const predictedConsumptionKwh = await getConsumptionPrediction(r);
      return { householdId: r.householdId, predictedSolarKwh, predictedConsumptionKwh };
    }
    const [predictedSolarKwh, predictedConsumptionKwh] = await Promise.all([getSolarPrediction(r), getConsumptionPrediction(r)]);
    return { householdId: r.householdId, predictedSolarKwh, predictedConsumptionKwh };
  });
}


export async function checkMlService(probeRow?: HouseholdRow): Promise<{ up: boolean; modelsLoaded: boolean; activeModel?: string; error?: string }> {
  try {
    await http.get("/health");
  } catch (e) {
    return { up: false, modelsLoaded: false, error: toMlError(e).message };
  }
  if (!probeRow) return { up: true, modelsLoaded: false, error: "no probe row available" };
  try {
    // Use the fallback-aware path: this will silently downgrade if the primary model needs
    // history that isn't available for the probe row (e.g. ensemble on day 1).
    const solar = await tryWithFallback(
      "solar",
      probeRow,
      (m, h) => solarBody(probeRow, m, h),
      "/predict/solar",
      "predicted_solar_generation_kwh"
    );
    await tryWithFallback(
      "consumption",
      probeRow,
      (m, h) => consumptionBody(probeRow, m, h),
      "/predict/consumption",
      "predicted_consumption_kwh"
    );
    return { up: true, modelsLoaded: true, activeModel: solar.modelUsed };
  } catch (e) {
    return { up: true, modelsLoaded: false, error: (e as Error).message };
  }
}



