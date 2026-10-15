import { describe, expect, it } from "vitest";
import { matchTrades } from "./matching.service";


const opts = { minKwh: 0.5, priceMilli: 6000 };


describe("matchTrades", () => {
  it("matches the largest seller with the largest buyer and trades the min", () => {
    const trades = matchTrades(
      [
        { householdId: 1, surplusMilli: 3000 },
        { householdId: 2, surplusMilli: 1000 },
        { householdId: 3, surplusMilli: -2000 },
        { householdId: 4, surplusMilli: -2500 },
      ],
      opts
    );
    expect(trades).toEqual([
      { sellerId: 1, buyerId: 4, kWhMilli: 2500, pricePerKwhMilli: 6000 },
      // seller 1 has 500 left, so seller 2 (1000) is now the largest
      { sellerId: 2, buyerId: 3, kWhMilli: 1000, pricePerKwhMilli: 6000 },
      { sellerId: 1, buyerId: 3, kWhMilli: 500, pricePerKwhMilli: 6000 },
    ]);
  });


  it("conserves energy: traded <= min(total supply, total demand) and no household is over-committed", () => {
    const entries = Array.from({ length: 25 }, (_, i) => ({
      householdId: i,
      surplusMilli: Math.round(Math.sin(i * 1.7) * 4000),
    }));
    const trades = matchTrades(entries, opts);
    const sold = new Map<number, number>();
    const bought = new Map<number, number>();
    for (const t of trades) {
      sold.set(t.sellerId, (sold.get(t.sellerId) ?? 0) + t.kWhMilli);
      bought.set(t.buyerId, (bought.get(t.buyerId) ?? 0) + t.kWhMilli);
      expect(t.sellerId).not.toBe(t.buyerId);
      expect(t.kWhMilli).toBeGreaterThanOrEqual(1);
    }
    for (const e of entries) {
      if (e.surplusMilli > 0) expect(sold.get(e.householdId) ?? 0).toBeLessThanOrEqual(e.surplusMilli);
      else expect(bought.get(e.householdId) ?? 0).toBeLessThanOrEqual(-e.surplusMilli);
      // a household is never both a seller and a buyer
      expect((sold.get(e.householdId) ?? 0) * (bought.get(e.householdId) ?? 0)).toBe(0);
    }
  });


  it("uses MIN_KWH as an eligibility threshold only", () => {
    const trades = matchTrades(
      [
        { householdId: 1, surplusMilli: 400 }, // below threshold -> not a seller
        { householdId: 2, surplusMilli: 700 }, // eligible; trades its full 700
        { householdId: 3, surplusMilli: -5000 },
      ],
      opts
    );
    expect(trades).toEqual([{ sellerId: 2, buyerId: 3, kWhMilli: 700, pricePerKwhMilli: 6000 }]);
  });


  it("is deterministic: ties go to the lower household id", () => {
    const entries = [
      { householdId: 7, surplusMilli: 2000 },
      { householdId: 3, surplusMilli: 2000 },
      { householdId: 9, surplusMilli: -2000 },
    ];
    expect(matchTrades(entries, opts)[0].sellerId).toBe(3);
    expect(matchTrades([...entries].reverse(), opts)).toEqual(matchTrades(entries, opts));
  });


  it("handles no sellers, no buyers, and empty input", () => {
    expect(matchTrades([], opts)).toEqual([]);
    expect(matchTrades([{ householdId: 1, surplusMilli: 5000 }], opts)).toEqual([]);
    expect(matchTrades([{ householdId: 1, surplusMilli: -5000 }], opts)).toEqual([]);
  });
});



