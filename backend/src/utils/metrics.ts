export interface MaeRow {
  predictedSolarKwh: number;
  actualSolarKwh: number;
  predictedConsumptionKwh: number;
  actualConsumptionKwh: number;
  isSolarAnomaly: boolean;
  isConsumptionAnomaly: boolean;
}


export interface MaeResult {
  mae: number | null;
  maeExcludingAnomalies: number | null;
  n: number;
  nAnomalies: number;
}


function mae(errors: number[]): number | null {
  return errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null;
}


function maeFor(rows: MaeRow[], pred: keyof MaeRow, act: keyof MaeRow, flag: keyof MaeRow): MaeResult {
  const all: number[] = [];
  const clean: number[] = [];
  for (const r of rows) {
    const err = Math.abs((r[pred] as number) - (r[act] as number));
    all.push(err);
    if (!r[flag]) clean.push(err);
  }
  return {
    mae: mae(all),
    maeExcludingAnomalies: mae(clean),
    n: all.length,
    nAnomalies: all.length - clean.length,
  };
}


export function computeMae(rows: MaeRow[]): { solar: MaeResult; consumption: MaeResult } {
  return {
    solar: maeFor(rows, "predictedSolarKwh", "actualSolarKwh", "isSolarAnomaly"),
    consumption: maeFor(rows, "predictedConsumptionKwh", "actualConsumptionKwh", "isConsumptionAnomaly"),
  };
}


// Anomaly hour: solar == 0 during daylight, or consumption == 0.
// "Daylight" is derived from the dataset's reference-only solar_irradiance column
// (never a model feature). The small threshold avoids flagging dawn/dusk hours where
// the generator forces output to 0 below its own irradiance cutoff.
export const DAYLIGHT_IRRADIANCE_MIN = 0.02;


export function isSolarAnomalyHour(solarIrradiance: number, actualSolarKwh: number): boolean {
  return solarIrradiance > DAYLIGHT_IRRADIANCE_MIN && actualSolarKwh === 0;
}


export function isConsumptionAnomalyHour(actualConsumptionKwh: number): boolean {
  return actualConsumptionKwh === 0;
}



