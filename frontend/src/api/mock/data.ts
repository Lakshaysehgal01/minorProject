// Deterministic fake backend state for VITE_USE_MOCK=true. Mirrors the real backend's definitions
// (greedy matching, savings, anomaly flags) so the UI behaves like the real thing.
import type {
  ActiveModels,
  DailySummary,
  ForecastResponse,
  Household,
  PredictionHour,
  SlotSummary,
  Stats,
  SummaryBlocks,
  Tier,
  Trade,
} from "../../types/api";


const HOUR = 3_600_000;
const DAY = 24 * HOUR;
export const SIM_START = Date.UTC(2026, 9, 20);
export const SIM_END = Date.UTC(2027, 0, 1);
export const DATASET_START = Date.UTC(2026, 0, 1);
const TOTAL_SLOTS = (SIM_END - SIM_START) / HOUR;
const PRICES = { grid: 8, p2p: 6, feedIn: 3 };
const MIN_KWH = 0.5;


// Tiers and panel sizes copied from the real generated dataset.
const HH: { tier: Tier; kw: number }[] = [
  ["small", 9.74], ["medium", 6.39], ["small", 10.59], ["medium", 8.97], ["medium", 2.94],
  ["small", 11.76], ["small", 9.61], ["small", 9.86], ["medium", 3.28], ["large", 6.5],
  ["large", 5.71], ["small", 11.27], ["small", 8.44], ["large", 10.23], ["small", 6.43],
  ["large", 4.27], ["medium", 7.55], ["large", 2.64], ["medium", 10.28], ["medium", 8.32],
  ["small", 9.58], ["small", 5.55], ["medium", 11.71], ["medium", 10.93], ["medium", 9.78],
].map(([tier, kw]) => ({ tier: tier as Tier, kw: kw as number }));


export const N_HOUSEHOLDS = HH.length;


export const MOCK_MODELS: ActiveModels = {
  solar: { model: "lightgbm", r2: 0.9578, source: "auto" },
  consumption: { model: "lightgbm", r2: 0.9102, source: "auto" },
};


// Two days are already simulated so the demo opens populated.
let cursorMs = Date.UTC(2026, 9, 22);
export const getCursor = () => cursorMs;
export const setCursor = (ms: number) => {
  cursorMs = ms;
};


function hash(...n: number[]): number {
  let h = 2166136261;
  for (const x of n) {
    h ^= (x | 0) + 0x9e3779b9 + (h << 6) + (h >>> 2);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 15;
  h = Math.imul(h, 2246822507);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}


const r4 = (x: number) => Math.round(x * 10000) / 10000;


export interface Rec {
  hh: number;
  slotMs: number;
  hour: number;
  predSolar: number;
  actSolar: number;
  predCons: number;
  actCons: number;
  surplus: number;
  solarAnom: boolean;
  consAnom: boolean;
}


const recCache = new Map<string, Rec>();


export function makeRec(hh: number, slotMs: number): Rec {
  const key = `${hh}:${slotMs}`;
  const hit = recCache.get(key);
  if (hit) return hit;
  const hour = new Date(slotMs).getUTCHours();
  const dayIdx = Math.floor(slotMs / DAY);
  const { tier, kw } = HH[hh];
  const sun = hour >= 6 && hour <= 18 ? Math.sin((Math.PI * (hour - 6)) / 12) : 0;
  const cloud = 0.55 + 0.4 * hash(dayIdx, 7);
  const solarBase = kw * 0.8 * sun * cloud;
  const solarFault = sun > 0.2 && hash(hh, slotMs, 2) < 0.012;
  const actSolar = solarFault ? 0 : solarBase * (1 + (hash(hh, slotMs, 1) - 0.5) * 0.1);
  const predSolar = Math.max(0, solarBase * (1 + (hash(hh, slotMs, 3) - 0.5) * 0.12));


  const base = { small: 0.3, medium: 0.5, large: 0.9 }[tier];
  const peak = { small: 0.7, medium: 1, large: 1.5 }[tier];
  const shape =
    base +
    peak * (1.2 * Math.exp(-((hour - 8) ** 2) / (2 * 1.5 ** 2)) + 1.8 * Math.exp(-((hour - 19) ** 2) / (2 * 2 ** 2)));
  const consOutage = hash(hh, slotMs, 4) < 0.005;
  const actCons = consOutage ? 0 : shape * (0.9 + 0.2 * hash(hh, slotMs, 5));
  const predCons = shape * (1 + (hash(hh, slotMs, 6) - 0.5) * 0.16);


  const rec: Rec = {
    hh,
    slotMs,
    hour,
    predSolar: r4(predSolar),
    actSolar: r4(actSolar),
    predCons: r4(predCons),
    actCons: r4(actCons),
    surplus: r4(predSolar - predCons),
    solarAnom: solarFault,
    consAnom: consOutage,
  };
  recCache.set(key, rec);
  return rec;
}


export const slotRecs = (slotMs: number): Rec[] => Array.from({ length: N_HOUSEHOLDS }, (_, h) => makeRec(h, slotMs));


interface Match {
  seller: number;
  buyer: number;
  milli: number;
}


function greedy(entries: { id: number; milli: number }[]): Match[] {
  const minMilli = MIN_KWH * 1000;
  const sellers = entries.filter((e) => e.milli > minMilli).map((e) => ({ id: e.id, rem: e.milli }));
  const buyers = entries.filter((e) => -e.milli > minMilli).map((e) => ({ id: e.id, rem: -e.milli }));
  const pick = (side: { id: number; rem: number }[]) =>
    side.reduce<{ id: number; rem: number } | undefined>(
      (best, s) => (s.rem > 0 && (!best || s.rem > best.rem || (s.rem === best.rem && s.id < best.id)) ? s : best),
      undefined
    );
  const out: Match[] = [];
  for (;;) {
    const s = pick(sellers);
    const b = pick(buyers);
    if (!s || !b) break;
    const milli = Math.min(s.rem, b.rem);
    out.push({ seller: s.id, buyer: b.id, milli });
    s.rem -= milli;
    b.rem -= milli;
  }
  return out;
}


const hex = (...n: number[]) =>
  Array.from({ length: 8 }, (_, i) => Math.floor(hash(...n, i) * 0xffffffff).toString(16).padStart(8, "0")).join("");


let tradeCacheCursor = -1;
let tradeCache: Trade[] = [];


// Chronological list of every trade up to the cursor. tradeId counts only confirmed trades.
export function allTrades(): Trade[] {
  if (tradeCacheCursor === cursorMs) return tradeCache;
  const trades: Trade[] = [];
  let nextTradeId = 0;
  for (let slotMs = SIM_START; slotMs < cursorMs; slotMs += HOUR) {
    const matches = greedy(slotRecs(slotMs).map((r) => ({ id: r.hh, milli: Math.round(r.surplus * 1000) })));
    matches.forEach((m, i) => {
      const failed = hash(slotMs, m.seller, m.buyer, 9) < 0.03;
      const iso = new Date(slotMs).toISOString();
      trades.push({
        _id: hex(slotMs, i).slice(0, 24),
        ...(failed ? {} : { tradeId: nextTradeId++, txHash: `0x${hex(slotMs, m.seller, m.buyer)}` }),
        sellerId: m.seller,
        buyerId: m.buyer,
        kWh: m.milli / 1000,
        kWhMilli: m.milli,
        pricePerKwh: PRICES.p2p,
        costMilli: Math.floor((m.milli * PRICES.p2p * 1000) / 1000),
        status: failed ? "failed" : "confirmed",
        ...(failed ? { error: "execution reverted (mock failure)" } : {}),
        slotTimestamp: iso,
        createdAt: new Date(slotMs + 5000 + i * 300).toISOString(),
      });
    });
  }
  tradeCache = trades;
  tradeCacheCursor = cursorMs;
  return trades;
}


export function households(): Household[] {
  const energy = new Array<number>(N_HOUSEHOLDS).fill(0);
  const token = new Array<number>(N_HOUSEHOLDS).fill(0);
  for (const t of allTrades()) {
    if (t.status !== "confirmed") continue;
    energy[t.sellerId] -= t.kWhMilli;
    energy[t.buyerId] += t.kWhMilli;
    token[t.buyerId] -= t.costMilli;
    token[t.sellerId] += t.costMilli;
  }
  return HH.map((h, i) => ({
    householdId: i,
    tier: h.tier,
    panelCapacityKw: h.kw,
    energyBalanceMilli: energy[i],
    tokenBalanceMilli: token[i],
    energyBalanceKwh: energy[i] / 1000,
    tokenBalance: token[i] / 1000,
  }));
}


export function buildBlocks(recs: Rec[], trades: Trade[]): SummaryBlocks {
  let predSolar = 0, actSolar = 0, predCons = 0, actCons = 0, predDeficit = 0, actDeficit = 0, baseline = 0;
  const slots = new Set<number>();
  const actSurplus = new Map<string, number>();
  const bySlot = new Map<number, { id: number; milli: number }[]>();
  for (const r of recs) {
    slots.add(r.slotMs);
    predSolar += r.predSolar; actSolar += r.actSolar; predCons += r.predCons; actCons += r.actCons;
    if (-r.surplus > MIN_KWH) predDeficit += -r.surplus;
    const net = r.actSolar - r.actCons;
    actDeficit += Math.max(0, -net);
    baseline += net < 0 ? -net * PRICES.grid : -net * PRICES.feedIn;
    actSurplus.set(`${r.slotMs}|${r.hh}`, Math.max(net, 0));
    const list = bySlot.get(r.slotMs) ?? [];
    list.push({ id: r.hh, milli: Math.round(net * 1000) });
    bySlot.set(r.slotMs, list);
  }
  const confirmed = trades.filter((t) => t.status === "confirmed");
  const failed = trades.length - confirmed.length;
  const traded = confirmed.reduce((a, t) => a + t.kWh, 0);
  const sold = new Map<string, number>();
  for (const t of confirmed) {
    const k = `${Date.parse(t.slotTimestamp)}|${t.sellerId}`;
    sold.set(k, (sold.get(k) ?? 0) + t.kWh);
  }
  let undelivered = 0;
  for (const [k, v] of sold) undelivered += Math.max(0, v - (actSurplus.get(k) ?? 0));
  const gross = traded * (PRICES.grid - PRICES.feedIn);
  let oracleMilli = 0;
  for (const entries of bySlot.values()) for (const m of greedy(entries)) oracleMilli += m.milli;


  const mae = (rows: Rec[], p: (r: Rec) => number, a: (r: Rec) => number, anom: (r: Rec) => boolean) => {
    const all = rows.map((r) => Math.abs(p(r) - a(r)));
    const clean = rows.filter((r) => !anom(r)).map((r) => Math.abs(p(r) - a(r)));
    const avg = (x: number[]) => (x.length ? x.reduce((s, v) => s + v, 0) / x.length : null);
    return { mae: avg(all), maeExcludingAnomalies: avg(clean), n: all.length, nAnomalies: all.length - clean.length };
  };


  return {
    slotsCovered: slots.size,
    solar: { predictedKwh: predSolar, actualKwh: actSolar },
    consumption: { predictedKwh: predCons, actualKwh: actCons },
    trading: {
      tradedKwh: traded,
      tradeCount: confirmed.length,
      failedTradeCount: failed,
      predictedDeficitKwh: predDeficit,
      actualDeficitKwh: actDeficit,
      percentDeficitMetP2P: predDeficit > 0 ? (traded / predDeficit) * 100 : null,
      percentActualDeficitMetP2P: actDeficit > 0 ? (traded / actDeficit) * 100 : null,
    },
    savings: {
      tradedKwh: traded,
      buyerSavings: (PRICES.grid - PRICES.p2p) * traded,
      sellerGains: (PRICES.p2p - PRICES.feedIn) * traded,
      grossSavings: gross,
      gridOnlyBaselineCost: baseline,
      p2pCost: baseline - gross,
      undeliveredKwh: undelivered,
      realizedSavings: gross - undelivered * (PRICES.grid - PRICES.p2p),
      perfectForesightTradedKwh: oracleMilli / 1000,
      perfectForesightSavings: (oracleMilli / 1000) * (PRICES.grid - PRICES.feedIn),
    },
    predictionError: {
      solar: mae(recs, (r) => r.predSolar, (r) => r.actSolar, (r) => r.solarAnom),
      consumption: mae(recs, (r) => r.predCons, (r) => r.actCons, (r) => r.consAnom),
    },
  };
}


const isoDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);


export function dayRecs(dateMs: number): Rec[] {
  const out: Rec[] = [];
  for (let s = dateMs; s < dateMs + DAY && s < cursorMs; s += HOUR) if (s >= SIM_START) out.push(...slotRecs(s));
  return out;
}


export function dailySummary(dateMs: number): DailySummary | null {
  const recs = dayRecs(dateMs);
  if (!recs.length) return null;
  const trades = allTrades().filter((t) => Date.parse(t.slotTimestamp) >= dateMs && Date.parse(t.slotTimestamp) < dateMs + DAY);
  const blocks = buildBlocks(recs, trades);
  return { date: isoDay(dateMs), complete: blocks.slotsCovered === 24, ...blocks };
}


export function stats(): Stats {
  const recs: Rec[] = [];
  for (let s = SIM_START; s < cursorMs; s += HOUR) recs.push(...slotRecs(s));
  const trades = allTrades();
  return {
    simulation: {
      cursor: new Date(cursorMs).toISOString(),
      slotsCompleted: (cursorMs - SIM_START) / HOUR,
      totalSlots: TOTAL_SLOTS,
      complete: cursorMs >= SIM_END,
      contractAddress: "0x5FbDB2315678afecb367f032d93F642f64180aa3",
      chainId: 31337,
    },
    onChainTradeCount: trades.filter((t) => t.status === "confirmed").length,
    ...buildBlocks(recs, trades),
  };
}


export function slotSummary(slotMs: number): SlotSummary {
  const trades = allTrades().filter((t) => Date.parse(t.slotTimestamp) === slotMs);
  const confirmed = trades.filter((t) => t.status === "confirmed");
  let surplus = 0, deficit = 0;
  for (const r of slotRecs(slotMs)) (r.surplus > 0 ? (surplus += r.surplus) : (deficit += -r.surplus));
  return {
    slot: new Date(slotMs).toISOString(),
    tradesConfirmed: confirmed.length,
    tradesFailed: trades.length - confirmed.length,
    kWhTraded: confirmed.reduce((a, t) => a + t.kWh, 0),
    predictedSurplusKwh: surplus,
    predictedDeficitKwh: deficit,
    nextCursor: new Date(slotMs + HOUR).toISOString(),
  };
}


export function predictionHours(hh: number, dateMs: number): PredictionHour[] {
  const out: PredictionHour[] = [];
  for (let s = dateMs; s < dateMs + DAY && s < cursorMs; s += HOUR) {
    if (s < SIM_START) continue;
    const r = makeRec(hh, s);
    out.push({
      slotTimestamp: new Date(s).toISOString(),
      householdId: hh,
      predictedSolarKwh: r.predSolar,
      predictedConsumptionKwh: r.predCons,
      actualSolarKwh: r.actSolar,
      actualConsumptionKwh: r.actCons,
      surplusKwh: r.surplus,
      modelUsed: "lightgbm",
      isSolarAnomaly: r.solarAnom,
      isConsumptionAnomaly: r.consAnom,
    });
  }
  return out;
}


export function forecast(hh: number, dateMs: number): ForecastResponse {
  const dayIdx = Math.floor(dateMs / DAY);
  return {
    householdId: hh,
    date: isoDay(dateMs),
    modelUsed: "lightgbm",
    models: MOCK_MODELS,
    inSample: dateMs < SIM_START,
    note:
      "Weather inputs are replayed from the dataset's own weather columns (the real observed weather for that hour). " +
      "A real deployment would feed a weather FORECAST API here, so these predictions are optimistic compared to production.",
    hours: Array.from({ length: 24 }, (_, hour) => {
      const r = makeRec(hh, dateMs + hour * HOUR);
      return {
        slotTimestamp: new Date(dateMs + hour * HOUR).toISOString(),
        hour,
        predictedSolarKwh: r.predSolar,
        predictedConsumptionKwh: r.predCons,
        predictedSurplusKwh: r.surplus,
        weatherInput: {
          temperature: Math.round((24 + 8 * Math.sin((Math.PI * (hour - 8)) / 12) + hash(dayIdx, 3) * 3) * 100) / 100,
          cloudCover: Math.round(hash(dayIdx, 7) * 100),
          humidity: Math.round(40 + hash(dayIdx, hour) * 30),
          windSpeed: Math.round(hash(dayIdx, hour, 1) * 60) / 10,
        },
      };
    }),
  };
}



