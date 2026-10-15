import { describe, expect, it, vi } from "vitest";
import type { HouseholdRow } from "./dataSource.service";


const HOUR = 3_600_000;
const T = Date.UTC(2026, 9, 20, 5);


const row = (ts: number, solar: number, cons: number): HouseholdRow => ({
  householdId: 3, timestampMs: ts, hour: new Date(ts).getUTCHours(), dayOfWeek: 1, dayOfYear: 293, month: 10,
  isHoliday: 0, tier: "medium", tierEncoded: 1, temperature: 20, cloudCover: 10, humidity: 50, windSpeed: 2,
  solarIrradiance: 0.1, panelCapacityKw: 6, actualSolarKwh: solar, actualConsumptionKwh: cons,
});


vi.mock("./dataSource.service", () => ({
  getHistory: (_id: number, ts: number) => {
    if (ts < Date.UTC(2026, 0, 2)) return null;
    return Array.from({ length: 24 }, (_, i) => row(ts - (24 - i) * HOUR, i, i / 10));
  },
}));


import { consumptionBody, solarBody } from "./mlClient.service";


describe("request bodies", () => {
  it("single-row models send just the hour's own features", () => {
    const body = solarBody(row(T, 9, 9), "lightgbm") as Record<string, unknown>;
    expect(body.hour).toBe(5);
    expect(body).not.toHaveProperty("rows");
  });


  it("lstm/ensemble send the 24 PAST hours (oldest first) with their observed targets, never the target hour", () => {
    const body = solarBody(row(T, 999, 999), "ensemble") as { rows: Record<string, number>[] };
    expect(body.rows).toHaveLength(24);
    expect(body.rows[0].solar_generation_kwh).toBe(0);
    expect(body.rows[23].solar_generation_kwh).toBe(23);
    expect(body.rows.some((r) => r.solar_generation_kwh === 999)).toBe(false);
    expect(body.rows[23].hour).toBe(4); // the hour before the 05:00 target


    const c = consumptionBody(row(T, 1, 1), "lstm") as { rows: Record<string, number>[] };
    expect(c.rows[23].consumption_kwh).toBeCloseTo(2.3);
  });


  it("explains when there is not enough history", () => {
    expect(() => solarBody(row(Date.UTC(2026, 0, 1, 0), 0, 0), "lstm")).toThrow(/24 hours of history/);
  });
});



