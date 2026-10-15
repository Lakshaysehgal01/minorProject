import { describe, expect, it } from "vitest";
import { computeSavings } from "./savings.service";


const prices = { grid: 8, p2p: 6, feedIn: 3 };
const slot = Date.UTC(2026, 9, 20, 12);


const pred = (householdId: number, solar: number, cons: number) => ({
  householdId,
  slotTimestamp: slot,
  actualSolarKwh: solar,
  actualConsumptionKwh: cons,
});


describe("computeSavings", () => {
  it("computes gross per-trade savings and the grid-only baseline on actuals", () => {
    // h1 surplus 3 kWh, h2 deficit 2 kWh
    const r = computeSavings([pred(1, 4, 1), pred(2, 0, 2)], [{ slotTimestamp: slot, sellerId: 1, buyerId: 2, kWh: 2 }], prices);
    expect(r.buyerSavings).toBeCloseTo((8 - 6) * 2);
    expect(r.sellerGains).toBeCloseTo((6 - 3) * 2);
    expect(r.grossSavings).toBeCloseTo(10);
    // baseline: h1 sells 3 @3 (-9), h2 buys 2 @8 (+16) = 7
    expect(r.gridOnlyBaselineCost).toBeCloseTo(7);
    expect(r.p2pCost).toBeCloseTo(7 - 10);
    expect(r.undeliveredKwh).toBe(0);
    expect(r.realizedSavings).toBeCloseTo(r.grossSavings);
  });


  it("penalises energy sold but not actually available (e.g. inverter fault)", () => {
    // seller was predicted to have surplus but actually generated 0 (and consumed 1)
    const r = computeSavings([pred(1, 0, 1), pred(2, 0, 2)], [{ slotTimestamp: slot, sellerId: 1, buyerId: 2, kWh: 2 }], prices);
    expect(r.undeliveredKwh).toBeCloseTo(2);
    expect(r.realizedSavings).toBeCloseTo(r.grossSavings - 2 * (8 - 6));
  });


  it("perfect foresight matches on actuals and bounds realised savings when forecasts are wrong", () => {
    const preds = [pred(1, 4, 1), pred(2, 0, 2), pred(3, 0, 3)];
    const r = computeSavings(preds, [], prices);
    // supply 3, demand 5 -> oracle trades 3 kWh at (GRID - FEEDIN)
    expect(r.perfectForesightTradedKwh).toBeCloseTo(3);
    expect(r.perfectForesightSavings).toBeCloseTo(3 * (8 - 3));
    expect(r.tradedKwh).toBe(0);
    expect(r.grossSavings).toBe(0);
  });


  it("handles zero actuals without producing NaN", () => {
    const r = computeSavings([pred(1, 0, 0), pred(2, 0, 0)], [], prices);
    for (const v of Object.values(r)) expect(Number.isFinite(v)).toBe(true);
  });
});



