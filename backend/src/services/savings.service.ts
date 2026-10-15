import { config } from "../config/env";
import { matchTrades } from "./matching.service";
import { kwhToMilli, priceToMilli } from "../utils/units";


export interface Prices {
  grid: number; // buy from utility
  p2p: number;
  feedIn: number; // sell to utility
}


export function getPrices(): Prices {
  return { grid: config.GRID_PRICE, p2p: config.P2P_PRICE_PER_KWH, feedIn: config.FEEDIN_PRICE };
}


export interface SavingsPrediction {
  householdId: number;
  slotTimestamp: Date | number;
  actualSolarKwh: number;
  actualConsumptionKwh: number;
}


export interface SavingsTrade {
  slotTimestamp: Date | number;
  sellerId: number;
  buyerId: number;
  kWh: number;
}


export interface SavingsResult {
  tradedKwh: number;
  buyerSavings: number; // (GRID - P2P) * kWh
  sellerGains: number; // (P2P - FEEDIN) * kWh
  grossSavings: number;
  gridOnlyBaselineCost: number; // actuals: every deficit bought at GRID, every surplus sold at FEEDIN
  p2pCost: number; // baseline minus gross savings (trades as committed)
  undeliveredKwh: number; // energy sold but not physically available per actuals
  realizedSavings: number; // gross minus the cost of re-buying undelivered energy from the grid
  perfectForesightTradedKwh: number;
  perfectForesightSavings: number; // same matcher run on actuals (upper bound)
}


const ms = (t: Date | number) => (typeof t === "number" ? t : t.getTime());


export function computeSavings(
  predictions: SavingsPrediction[],
  confirmedTrades: SavingsTrade[],
  prices: Prices = getPrices(),
  minKwh: number = config.MIN_KWH
): SavingsResult {
  let tradedKwh = 0;
  for (const t of confirmedTrades) tradedKwh += t.kWh;
  const buyerSavings = (prices.grid - prices.p2p) * tradedKwh;
  const sellerGains = (prices.p2p - prices.feedIn) * tradedKwh;
  const grossSavings = buyerSavings + sellerGains;


  let baseline = 0;
  const actualSurplus = new Map<string, number>();
  const bySlot = new Map<number, { householdId: number; surplusMilli: number }[]>();
  for (const p of predictions) {
    const net = p.actualSolarKwh - p.actualConsumptionKwh;
    baseline += net < 0 ? -net * prices.grid : -net * prices.feedIn;
    const slot = ms(p.slotTimestamp);
    actualSurplus.set(`${slot}|${p.householdId}`, Math.max(net, 0));
    const list = bySlot.get(slot);
    const entry = { householdId: p.householdId, surplusMilli: kwhToMilli(net) };
    if (list) list.push(entry);
    else bySlot.set(slot, [entry]);
  }


  const sold = new Map<string, number>();
  for (const t of confirmedTrades) {
    const key = `${ms(t.slotTimestamp)}|${t.sellerId}`;
    sold.set(key, (sold.get(key) ?? 0) + t.kWh);
  }
  let undeliveredKwh = 0;
  for (const [key, kwh] of sold) {
    undeliveredKwh += Math.max(0, kwh - (actualSurplus.get(key) ?? 0));
  }
  const realizedSavings = grossSavings - undeliveredKwh * (prices.grid - prices.p2p);


  let oracleMilli = 0;
  for (const entries of bySlot.values()) {
    for (const t of matchTrades(entries, { minKwh, priceMilli: priceToMilli(prices.p2p) })) oracleMilli += t.kWhMilli;
  }
  const perfectForesightTradedKwh = oracleMilli / 1000;


  return {
    tradedKwh,
    buyerSavings,
    sellerGains,
    grossSavings,
    gridOnlyBaselineCost: baseline,
    p2pCost: baseline - grossSavings,
    undeliveredKwh,
    realizedSavings,
    perfectForesightTradedKwh,
    perfectForesightSavings: perfectForesightTradedKwh * (prices.grid - prices.feedIn),
  };
}



