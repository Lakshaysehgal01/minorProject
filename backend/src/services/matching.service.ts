import { config } from "../config/env";
import { priceToMilli } from "../utils/units";


export interface MatchEntry {
  householdId: number;
  surplusMilli: number; // predicted solar - predicted consumption, signed, in milli-kWh
}


export interface MatchedTrade {
  sellerId: number;
  buyerId: number;
  kWhMilli: number;
  pricePerKwhMilli: number;
}


// Flat P2P price. Isolated so dynamic pricing can replace it later.
export function getP2PPriceMilli(): number {
  return priceToMilli(config.P2P_PRICE_PER_KWH);
}


interface Side {
  householdId: number;
  remaining: number;
}


// Largest remaining wins; ties go to the lower household id so runs are deterministic.
function pickLargest(side: Side[]): Side | undefined {
  let best: Side | undefined;
  for (const s of side) {
    if (s.remaining <= 0) continue;
    if (!best || s.remaining > best.remaining || (s.remaining === best.remaining && s.householdId < best.householdId)) {
      best = s;
    }
  }
  return best;
}


// MIN_KWH is an eligibility threshold only: an eligible household trades its full surplus/deficit.
export function matchTrades(
  entries: MatchEntry[],
  opts: { minKwh?: number; priceMilli?: number } = {}
): MatchedTrade[] {
  const minMilli = Math.round((opts.minKwh ?? config.MIN_KWH) * 1000);
  const pricePerKwhMilli = opts.priceMilli ?? getP2PPriceMilli();


  const sellers: Side[] = entries
    .filter((e) => e.surplusMilli > minMilli)
    .map((e) => ({ householdId: e.householdId, remaining: e.surplusMilli }));
  const buyers: Side[] = entries
    .filter((e) => -e.surplusMilli > minMilli)
    .map((e) => ({ householdId: e.householdId, remaining: -e.surplusMilli }));


  const trades: MatchedTrade[] = [];
  for (;;) {
    const seller = pickLargest(sellers);
    const buyer = pickLargest(buyers);
    if (!seller || !buyer) break;
    const kWhMilli = Math.min(seller.remaining, buyer.remaining);
    if (kWhMilli < 1) break;
    trades.push({ sellerId: seller.householdId, buyerId: buyer.householdId, kWhMilli, pricePerKwhMilli });
    seller.remaining -= kWhMilli;
    buyer.remaining -= kWhMilli;
  }
  return trades;
}



