import fs from "fs";
import { parse } from "csv-parse";
import { config } from "../config/env";
import { httpError } from "../utils/errors";
import { parseNaiveUtc } from "../utils/time";


export const EXPECTED_HOUSEHOLDS = 25;


export type Tier = "small" | "medium" | "large";


export interface HouseholdRow {
  householdId: number;
  timestampMs: number;
  hour: number;
  dayOfWeek: number;
  dayOfYear: number;
  month: number;
  isHoliday: number;
  tier: Tier;
  tierEncoded: number;
  temperature: number;
  cloudCover: number;
  humidity: number;
  windSpeed: number;
  solarIrradiance: number;
  panelCapacityKw: number;
  actualSolarKwh: number;
  actualConsumptionKwh: number;
}


export interface HouseholdInfo {
  householdId: number;
  tier: Tier;
  panelCapacityKw: number;
}


const REQUIRED_COLUMNS = [
  "timestamp", "hour", "day_of_week", "day_of_year", "month", "is_holiday", "household_id",
  "consumption_profile", "consumption_profile_encoded", "temperature", "cloud_cover", "humidity",
  "wind_speed", "solar_irradiance", "panel_capacity_kw", "solar_generation_kwh", "consumption_kwh",
];


const slots = new Map<number, HouseholdRow[]>();
let households: HouseholdInfo[] = [];
let loaded = false;


export async function loadData(csvPath: string = config.DATA_CSV_PATH): Promise<void> {
  if (loaded) return;
  if (!fs.existsSync(csvPath)) {
    throw new Error(
      `Dataset not found at ${csvPath}. Generate it with \`python data/generate_data.py\` in ml-service, or set DATA_CSV_PATH.`
    );
  }


  const parser = fs.createReadStream(csvPath).pipe(parse({ columns: true, skip_empty_lines: true }));
  let checkedColumns = false;


  for await (const r of parser as AsyncIterable<Record<string, string>>) {
    if (!checkedColumns) {
      const missing = REQUIRED_COLUMNS.filter((c) => !(c in r));
      if (missing.length) throw new Error(`Dataset is missing columns: ${missing.join(", ")}`);
      checkedColumns = true;
    }
    const timestampMs = parseNaiveUtc(r.timestamp);
    const row: HouseholdRow = {
      householdId: Number(r.household_id),
      timestampMs,
      hour: Number(r.hour),
      dayOfWeek: Number(r.day_of_week),
      dayOfYear: Number(r.day_of_year),
      month: Number(r.month),
      isHoliday: Number(r.is_holiday),
      tier: r.consumption_profile as Tier,
      tierEncoded: Number(r.consumption_profile_encoded),
      temperature: Number(r.temperature),
      cloudCover: Number(r.cloud_cover),
      humidity: Number(r.humidity),
      windSpeed: Number(r.wind_speed),
      solarIrradiance: Number(r.solar_irradiance),
      panelCapacityKw: Number(r.panel_capacity_kw),
      actualSolarKwh: Number(r.solar_generation_kwh),
      actualConsumptionKwh: Number(r.consumption_kwh),
    };
    const bucket = slots.get(timestampMs);
    if (bucket) bucket.push(row);
    else slots.set(timestampMs, [row]);
  }


  for (const [ts, rows] of slots) {
    if (rows.length !== EXPECTED_HOUSEHOLDS) {
      throw new Error(`Slot ${new Date(ts).toISOString()} has ${rows.length} rows, expected ${EXPECTED_HOUSEHOLDS}`);
    }
    rows.sort((a, b) => a.householdId - b.householdId);
  }


  const firstTs = Math.min(...slots.keys());
  households = slots.get(firstTs)!.map((r) => ({
    householdId: r.householdId,
    tier: r.tier,
    panelCapacityKw: r.panelCapacityKw,
  }));
  loaded = true;
}


export function isDataLoaded(): boolean {
  return loaded;
}


// Returns the 25 households' rows for that hour, or undefined if the dataset has no such slot.
export function getSlot(timestampMs: number): HouseholdRow[] | undefined {
  return slots.get(timestampMs);
}


// The 24 already-elapsed hours before `timestampMs` (oldest first) for one household, as needed by
// the lstm/ensemble models. Returns null when the dataset does not reach back far enough.
export function getHistory(householdId: number, timestampMs: number, hours = 24): HouseholdRow[] | null {
  const out: HouseholdRow[] = [];
  for (let k = hours; k >= 1; k--) {
    const row = slots.get(timestampMs - k * 3_600_000)?.find((r) => r.householdId === householdId);
    if (!row) return null;
    out.push(row);
  }
  return out;
}


export function getHouseholdInfos(): HouseholdInfo[] {
  if (!loaded) throw httpError(503, "Dataset not loaded");
  return households;
}



