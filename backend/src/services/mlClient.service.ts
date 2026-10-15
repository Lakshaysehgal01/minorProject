import axios, { AxiosError } from "axios";
import { config } from "../config/env";
import { HouseholdRow, getHistory } from "./dataSource.service";
import { getSelection, isSequenceModel } from "./modelSelection.service";
import { HttpError, httpError } from "../utils/errors";


const BATCH_SIZE = 25;


const http = axios.create({ baseURL: config.ML_SERVICE_URL, timeout: 5000 });


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
    const reason = ax.code === "ECONNABORTED" || ax.code === "ETIMEDOUT" ? "timed out after 5s" : "unreachable";
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
function historyOrThrow(row: HouseholdRow, model: string): HouseholdRow[] {
  const history = getHistory(row.householdId, row.timestampMs);
  if (!history) {
    throw httpError(
      422,
      `Model "${model}" needs 24 hours of history before ${new Date(row.timestampMs).toISOString()}, but the dataset does not go back that far.`
    );
  }
  return history;
}


export function solarBody(row: HouseholdRow, model: string) {
  if (!isSequenceModel(model)) return solarPayload(row);
  return { rows: historyOrThrow(row, model).map((r) => ({ ...solarPayload(r), solar_generation_kwh: r.actualSolarKwh })) };
}


export function consumptionBody(row: HouseholdRow, model: string) {
  if (!isSequenceModel(model)) return consumptionPayload(row);
  return { rows: historyOrThrow(row, model).map((r) => ({ ...consumptionPayload(r), consumption_kwh: r.actualConsumptionKwh })) };
}


export async function getSolarPrediction(row: HouseholdRow, model: string = getSelection("solar").model): Promise<number> {
  try {
    const res = await http.post("/predict/solar", solarBody(row, model), { params: { model } });
    return clamp(res.data?.predicted_solar_generation_kwh, "solar prediction");
  } catch (e) {
    throw (e as HttpError).status ? e : toMlError(e);
  }
}


export async function getConsumptionPrediction(row: HouseholdRow, model: string = getSelection("consumption").model): Promise<number> {
  try {
    const res = await http.post("/predict/consumption", consumptionBody(row, model), { params: { model } });
    return clamp(res.data?.predicted_consumption_kwh, "consumption prediction");
  } catch (e) {
    throw (e as HttpError).status ? e : toMlError(e);
  }
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


// Predict a set of households in parallel batches (the ML API is single-row only).
export async function predictRows(rows: HouseholdRow[]): Promise<RowPrediction[]> {
  const out: RowPrediction[] = [];
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const chunk = rows.slice(i, i + BATCH_SIZE);
    const results = await Promise.all(
      chunk.map(async (r) => {
        const [predictedSolarKwh, predictedConsumptionKwh] = await Promise.all([
          getSolarPrediction(r),
          getConsumptionPrediction(r),
        ]);
        return { householdId: r.householdId, predictedSolarKwh, predictedConsumptionKwh };
      })
    );
    out.push(...results);
  }
  return out;
}


export async function checkMlService(probeRow?: HouseholdRow): Promise<{ up: boolean; modelsLoaded: boolean; error?: string }> {
  try {
    await http.get("/health");
  } catch (e) {
    return { up: false, modelsLoaded: false, error: toMlError(e).message };
  }
  if (!probeRow) return { up: true, modelsLoaded: false, error: "no probe row available" };
  try {
    await Promise.all([getSolarPrediction(probeRow), getConsumptionPrediction(probeRow)]);
    return { up: true, modelsLoaded: true };
  } catch (e) {
    return { up: true, modelsLoaded: false, error: (e as Error).message };
  }
}



