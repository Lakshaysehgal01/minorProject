import { describe, expect, it } from "vitest";
import { computeMae, isConsumptionAnomalyHour, isSolarAnomalyHour } from "./metrics";
import { costMilli, kwhToMilli } from "./units";
import { parseDateKey, parseNaiveUtc } from "./time";


describe("computeMae", () => {
  it("reports MAE with and without anomaly hours", () => {
    const row = (ps: number, as: number, solarAnom: boolean) => ({
      predictedSolarKwh: ps,
      actualSolarKwh: as,
      predictedConsumptionKwh: 1,
      actualConsumptionKwh: 1,
      isSolarAnomaly: solarAnom,
      isConsumptionAnomaly: false,
    });
    const r = computeMae([row(2, 1, false), row(4, 0, true)]); // errors 1 and 4
    expect(r.solar.mae).toBeCloseTo(2.5);
    expect(r.solar.maeExcludingAnomalies).toBeCloseTo(1);
    expect(r.solar.nAnomalies).toBe(1);
    expect(r.consumption.mae).toBe(0);
  });


  it("returns null MAE for no rows", () => {
    expect(computeMae([]).solar.mae).toBeNull();
  });
});


describe("anomaly flags", () => {
  it("flags zero solar only in daylight, and any zero consumption", () => {
    expect(isSolarAnomalyHour(0.8, 0)).toBe(true);
    expect(isSolarAnomalyHour(0, 0)).toBe(false); // night
    expect(isSolarAnomalyHour(0.8, 2)).toBe(false);
    expect(isConsumptionAnomalyHour(0)).toBe(true);
    expect(isConsumptionAnomalyHour(0.3)).toBe(false);
  });
});


describe("units and time", () => {
  it("mirrors the contract's integer cost formula", () => {
    expect(kwhToMilli(2.5)).toBe(2500);
    expect(costMilli(2500, 6000)).toBe(15000);
    expect(costMilli(1, 1)).toBe(0);
  });


  it("parses naive timestamps as UTC regardless of host timezone", () => {
    expect(parseNaiveUtc("2026-10-20 00:00:00")).toBe(Date.UTC(2026, 9, 20));
    expect(parseDateKey("2026-10-20")).toBe(Date.UTC(2026, 9, 20));
    expect(parseDateKey("2026-02-30")).toBeNull();
    expect(parseDateKey("20-10-2026")).toBeNull();
  });
});



