import { getHouseholdInfos, getSlot } from "./dataSource.service";
import { modelLabel, predictRows } from "./mlClient.service";
import { getActiveModels } from "./modelSelection.service";
import { HOUR_MS, SIM_START_MS, parseDateKey } from "../utils/time";
import { httpError } from "../utils/errors";


export const FORECAST_NOTE =
  "Weather inputs are replayed from the dataset's own weather columns (the real observed weather for that hour). " +
  "A real deployment would feed a weather FORECAST API here, so these predictions are optimistic compared to production.";


// Predictions for one household and date, using the dataset's weather columns as input.
// No actuals are attached and nothing is persisted.
export async function getForecast(householdId: number, date: string) {
  const dayStart = parseDateKey(date);
  if (dayStart === null) throw httpError(400, "date must be a valid YYYY-MM-DD");
  if (!getHouseholdInfos().some((h) => h.householdId === householdId)) {
    throw httpError(404, `Unknown household ${householdId}`);
  }


  const rows = [];
  for (let h = 0; h < 24; h++) {
    const slot = getSlot(dayStart + h * HOUR_MS);
    const row = slot?.find((r) => r.householdId === householdId);
    if (!row) throw httpError(404, `Dataset has no weather data for ${date} (covers 2026-01-01 to 2026-12-31)`);
    rows.push(row);
  }


  const predicted = await predictRows(rows);
  return {
    householdId,
    date,
    modelUsed: modelLabel(),
    models: getActiveModels(), // which model produced each target's numbers
    inSample: dayStart < SIM_START_MS, // before the test period = part of the models' training data
    note: FORECAST_NOTE,
    hours: rows.map((r, i) => ({
      slotTimestamp: new Date(r.timestampMs).toISOString(),
      hour: r.hour,
      predictedSolarKwh: predicted[i].predictedSolarKwh,
      predictedConsumptionKwh: predicted[i].predictedConsumptionKwh,
      predictedSurplusKwh: predicted[i].predictedSolarKwh - predicted[i].predictedConsumptionKwh,
      weatherInput: {
        temperature: r.temperature,
        cloudCover: r.cloudCover,
        humidity: r.humidity,
        windSpeed: r.windSpeed,
      },
    })),
  };
}



