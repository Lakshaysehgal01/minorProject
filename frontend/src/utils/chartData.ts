import type { ForecastHour, ModelRow, PredictionHour } from "../types/api";
import { hourLabel, hourOf } from "./date";


export interface HourPoint {
  hour: number;
  label: string;
  predictedSolar?: number;
  actualSolar?: number;
  predictedConsumption?: number;
  actualConsumption?: number;
  predictedNet?: number;
  actualNet?: number;
  solarAnomaly: boolean;
  consumptionAnomaly: boolean;
}


// Always 24 points so the x-axis stays 00:00-23:00 even for a partially simulated day.
export function toHourSeries(hours: PredictionHour[]): HourPoint[] {
  const byHour = new Map(hours.map((h) => [hourOf(h.slotTimestamp), h]));
  return Array.from({ length: 24 }, (_, hour) => {
    const h = byHour.get(hour);
    return {
      hour,
      label: hourLabel(hour),
      predictedSolar: h?.predictedSolarKwh,
      actualSolar: h?.actualSolarKwh,
      predictedConsumption: h?.predictedConsumptionKwh,
      actualConsumption: h?.actualConsumptionKwh,
      predictedNet: h?.surplusKwh,
      actualNet: h ? h.actualSolarKwh - h.actualConsumptionKwh : undefined,
      solarAnomaly: h?.isSolarAnomaly ?? false,
      consumptionAnomaly: h?.isConsumptionAnomaly ?? false,
    };
  });
}


export interface ForecastPoint {
  hour: number;
  label: string;
  solar: number;
  consumption: number;
  net: number;
}


export function toForecastSeries(hours: ForecastHour[]): ForecastPoint[] {
  return hours.map((h) => ({
    hour: h.hour,
    label: hourLabel(h.hour),
    solar: h.predictedSolarKwh,
    consumption: h.predictedConsumptionKwh,
    net: h.predictedSurplusKwh,
  }));
}


export function forecastTotals(points: ForecastPoint[]) {
  return points.reduce(
    (t, p) => ({ solar: t.solar + p.solar, consumption: t.consumption + p.consumption, net: t.net + p.net }),
    { solar: 0, consumption: 0, net: 0 }
  );
}


const MODEL_LABELS: Record<string, string> = {
  LGBMRegressor: "LightGBM",
  XGBRegressor: "XGBoost",
  CatBoostRegressor: "CatBoost",
  RandomForestRegressor: "Random Forest",
  DecisionTreeRegressor: "Decision Tree",
  LinearRegression: "Linear Regression",
  LSTM: "LSTM",
  Ensemble: "Stacking Ensemble",
};


const SERVED_NAMES: Record<string, string> = {
  xgboost: "XGBRegressor",
  lightgbm: "LGBMRegressor",
  catboost: "CatBoostRegressor",
  lstm: "LSTM",
  ensemble: "Ensemble",
};


// Does this comparison-table row correspond to the model the backend is actually calling?
export function isActiveRow(csvName: string, active: string | undefined): boolean {
  return active !== undefined && SERVED_NAMES[active] === csvName;
}


export function modelDisplayName(servedName: string): string {
  return MODEL_LABELS[SERVED_NAMES[servedName] ?? ""] ?? servedName.toUpperCase().slice(0, 1) + servedName.slice(1);
}


export function modelLabel(name: string): string {
  const prophet = /^Prophet_hh(\d+)_(\w+)$/.exec(name);
  if (prophet) return `Prophet (household ${prophet[1]}, ${prophet[2]})*`;
  return MODEL_LABELS[name] ?? name;
}


export function isBenchmarkRow(name: string): boolean {
  return name.startsWith("Prophet_");
}


export function sortByR2(rows: ModelRow[]): ModelRow[] {
  return [...rows].sort((a, b) => b.R2 - a.R2);
}


// Start just below the worst model instead of 0 so differences stay visible.
export function r2Domain(rows: ModelRow[]): [number, number] {
  if (!rows.length) return [0, 1];
  const min = Math.min(...rows.map((r) => r.R2));
  return [Math.floor((min - 0.02) * 100) / 100, 1];
}



