import { Prediction, PredictionDoc } from "../models/prediction.model";
import { Trade } from "../models/trade.model";
import { SimulationState, SimulationStateDoc } from "../models/simulationState.model";
import { getSlot } from "./dataSource.service";
import { modelLabel, predictRows } from "./mlClient.service";
import { MatchedTrade, matchTrades } from "./matching.service";
import {
  executeTradeOnChain,
  findTradeOnChain,
  getChain,
  getOnChainTradeCount,
  verifyContractDeployed,
} from "./blockchain.service";
import { getDailySummary } from "./daily-summary.service";
import { DAY_MS, HOUR_MS, SIM_END_MS, SIM_START_MS, TOTAL_SLOTS, dateKey, startOfDay } from "../utils/time";
import { costMilli, kwhToMilli } from "../utils/units";
import { errorMessage, httpError } from "../utils/errors";
import { isConsumptionAnomalyHour, isSolarAnomalyHour } from "../utils/metrics";


export const MAX_HOURS_PER_RUN = 168;


export interface SlotSummary {
  slot: string;
  tradesConfirmed: number;
  tradesFailed: number;
  kWhTraded: number;
  predictedSurplusKwh: number;
  predictedDeficitKwh: number;
  nextCursor: string;
}


// ---- concurrency lock -------------------------------------------------------
let busy = false;


export function isBusy(): boolean {
  return busy;
}


async function withLock<T>(fn: () => Promise<T>): Promise<T> {
  if (busy) throw httpError(409, "Simulation busy: another run is in progress.");
  busy = true;
  try {
    return await fn();
  } finally {
    busy = false;
  }
}


// ---- state ------------------------------------------------------------------
async function getState(): Promise<SimulationStateDoc> {
  const state = await SimulationState.findOneAndUpdate(
    { key: "singleton" },
    { $setOnInsert: { key: "singleton", cursor: new Date(SIM_START_MS) } },
    { upsert: true, new: true }
  ).lean();
  return state as SimulationStateDoc;
}


export async function getSimulationStatus() {
  const state = await getState();
  const cursorMs = state.cursor.getTime();
  const slotsCompleted = Math.round((cursorMs - SIM_START_MS) / HOUR_MS);
  return {
    cursor: state.cursor.toISOString(),
    slotsCompleted,
    totalSlots: TOTAL_SLOTS,
    complete: cursorMs >= SIM_END_MS,
    contractAddress: state.contractAddress ?? null,
    chainId: state.chainId ?? null,
  };
}


// Chain and DB must agree. The Hardhat node is in-memory, so a node restart leaves Mongo ahead.
export async function checkSync(state?: SimulationStateDoc): Promise<{
  ok: boolean;
  reason?: string;
  onChainTradeCount?: number;
  dbConfirmedTrades?: number;
}> {
  const s = state ?? (await getState());
  const { address, chainId } = getChain();
  if (s.contractAddress && (s.contractAddress.toLowerCase() !== address.toLowerCase() || s.chainId !== chainId)) {
    return {
      ok: false,
      reason: `Simulation was bound to contract ${s.contractAddress} (chain ${s.chainId}) but the backend now points at ${address} (chain ${chainId}). POST /simulate/reset to start over.`,
    };
  }
  const [onChainTradeCount, dbConfirmedTrades] = await Promise.all([
    getOnChainTradeCount(),
    Trade.countDocuments({ status: "confirmed" }),
  ]);
  if (onChainTradeCount < dbConfirmedTrades) {
    return {
      ok: false,
      onChainTradeCount,
      dbConfirmedTrades,
      reason: `Chain has ${onChainTradeCount} trades but the DB has ${dbConfirmedTrades} confirmed. The Hardhat node was likely restarted. Redeploy, then POST /simulate/reset to clear the DB.`,
    };
  }
  return { ok: true, onChainTradeCount, dbConfirmedTrades };
}


async function prepareRun(): Promise<SimulationStateDoc> {
  let state = await getState();
  if (state.cursor.getTime() >= SIM_END_MS) {
    throw httpError(409, "Simulation complete: the whole test period (to 2026-12-31 23:00) has been replayed. POST /simulate/reset to start over.");
  }
  await verifyContractDeployed();
  const sync = await checkSync(state);
  if (!sync.ok) throw httpError(409, sync.reason ?? "Chain and database are out of sync");
  if (!state.contractAddress) {
    const { address, chainId } = getChain();
    state = (await SimulationState.findOneAndUpdate(
      { key: "singleton" },
      { $set: { contractAddress: address, chainId } },
      { new: true }
    ).lean()) as SimulationStateDoc;
  }
  return state;
}


// ---- slot processing --------------------------------------------------------
type PredLean = Pick<
  PredictionDoc,
  "householdId" | "predictedSolarKwh" | "predictedConsumptionKwh" | "surplusKwh"
>;


async function settleMatch(slot: Date, m: MatchedTrade): Promise<"confirmed" | "failed"> {
  const filter = { slotTimestamp: slot, sellerId: m.sellerId, buyerId: m.buyerId };
  const kWh = m.kWhMilli / 1000;
  const pricePerKwh = m.pricePerKwhMilli / 1000;


  const prior = await Trade.findOne(filter).lean();
  if (prior?.status === "confirmed") return "confirmed";
  if (prior?.status === "failed") return "failed";


  if (prior) {
    // `pending` left by an interrupted run: the tx may have landed. Check the chain before re-sending.
    const found = await findTradeOnChain(m.sellerId, m.buyerId, slot.getTime());
    if (found) {
      await Trade.updateOne(filter, { $set: { status: "confirmed", txHash: found.txHash, tradeId: found.tradeId } });
      return "confirmed";
    }
  } else {
    await Trade.create({
      ...filter,
      kWh,
      kWhMilli: m.kWhMilli,
      pricePerKwh,
      costMilli: costMilli(m.kWhMilli, m.pricePerKwhMilli),
      status: "pending",
    });
  }


  try {
    const { txHash, tradeId } = await executeTradeOnChain({
      sellerId: m.sellerId,
      buyerId: m.buyerId,
      kWh,
      pricePerKwh,
      slotTimestampMs: slot.getTime(),
    });
    await Trade.updateOne(filter, { $set: { status: "confirmed", txHash, tradeId } });
    return "confirmed";
  } catch (e) {
    await Trade.updateOne(filter, { $set: { status: "failed", error: errorMessage(e).slice(0, 500) } });
    return "failed";
  }
}


async function processSlot(cursor: Date): Promise<SlotSummary> {
  const rows = getSlot(cursor.getTime());
  if (!rows) throw httpError(500, `Dataset has no rows for slot ${cursor.toISOString()}`);


  // Predictions: reuse any already saved for this slot (idempotent after an interrupted run).
  const existing = await Prediction.find({ slotTimestamp: cursor }).lean();
  const byHousehold = new Map<number, PredLean>(existing.map((p) => [p.householdId, p]));
  const missing = rows.filter((r) => !byHousehold.has(r.householdId));
  if (missing.length) {
    const predicted = await predictRows(missing);
    const rowById = new Map(rows.map((r) => [r.householdId, r]));
    const docs: PredictionDoc[] = predicted.map((p) => {
      const r = rowById.get(p.householdId)!;
      return {
        householdId: p.householdId,
        slotTimestamp: cursor,
        predictedSolarKwh: p.predictedSolarKwh,
        predictedConsumptionKwh: p.predictedConsumptionKwh,
        actualSolarKwh: r.actualSolarKwh,
        actualConsumptionKwh: r.actualConsumptionKwh,
        surplusKwh: p.predictedSolarKwh - p.predictedConsumptionKwh,
        modelUsed: modelLabel(),
        isSolarAnomaly: isSolarAnomalyHour(r.solarIrradiance, r.actualSolarKwh),
        isConsumptionAnomaly: isConsumptionAnomalyHour(r.actualConsumptionKwh),
      };
    });
    await Prediction.bulkWrite(
      docs.map((d) => ({
        updateOne: {
          filter: { householdId: d.householdId, slotTimestamp: d.slotTimestamp },
          update: { $set: d },
          upsert: true,
        },
      }))
    );
    for (const d of docs) byHousehold.set(d.householdId, d);
  }


  const predictions = rows.map((r) => byHousehold.get(r.householdId)!);
  const matches = matchTrades(
    predictions.map((p) => ({ householdId: p.householdId, surplusMilli: kwhToMilli(p.surplusKwh) }))
  );


  let tradesConfirmed = 0;
  let tradesFailed = 0;
  let kWhTradedMilli = 0;
  for (const m of matches) {
    const outcome = await settleMatch(cursor, m); // sequential: nonce safety
    if (outcome === "confirmed") {
      tradesConfirmed++;
      kWhTradedMilli += m.kWhMilli;
    } else tradesFailed++;
  }


  const next = new Date(cursor.getTime() + HOUR_MS);
  const advanced = await SimulationState.updateOne({ key: "singleton", cursor }, { $set: { cursor: next } });
  if (advanced.modifiedCount !== 1) throw httpError(409, "Simulation cursor changed during the run");


  let predictedSurplusKwh = 0;
  let predictedDeficitKwh = 0;
  for (const p of predictions) {
    if (p.surplusKwh > 0) predictedSurplusKwh += p.surplusKwh;
    else predictedDeficitKwh += -p.surplusKwh;
  }


  return {
    slot: cursor.toISOString(),
    tradesConfirmed,
    tradesFailed,
    kWhTraded: kWhTradedMilli / 1000,
    predictedSurplusKwh,
    predictedDeficitKwh,
    nextCursor: next.toISOString(),
  };
}


// ---- public API -------------------------------------------------------------
export async function runSlot(): Promise<SlotSummary> {
  return withLock(async () => {
    const state = await prepareRun();
    return processSlot(state.cursor);
  });
}


export async function runHours(hours: number) {
  if (!Number.isInteger(hours) || hours < 1 || hours > MAX_HOURS_PER_RUN) {
    throw httpError(400, `hours must be an integer between 1 and ${MAX_HOURS_PER_RUN}`);
  }
  return withLock(async () => {
    let state = await prepareRun();
    const slots: SlotSummary[] = [];
    while (slots.length < hours && state.cursor.getTime() < SIM_END_MS) {
      const summary = await processSlot(state.cursor);
      slots.push(summary);
      state = { ...state, cursor: new Date(summary.nextCursor) };
    }
    return {
      slotsRun: slots.length,
      complete: state.cursor.getTime() >= SIM_END_MS,
      nextCursor: state.cursor.toISOString(),
      kWhTraded: slots.reduce((a, s) => a + s.kWhTraded, 0),
      tradesConfirmed: slots.reduce((a, s) => a + s.tradesConfirmed, 0),
      tradesFailed: slots.reduce((a, s) => a + s.tradesFailed, 0),
      slots,
    };
  });
}


// Runs until the cursor reaches the next 00:00 (24 slots when starting at midnight),
// so day summaries always line up with calendar days.
export async function runDay() {
  const result = await withLock(async () => {
    let state = await prepareRun();
    const dayStart = startOfDay(state.cursor.getTime());
    const dayEnd = dayStart + DAY_MS;
    const slots: SlotSummary[] = [];
    while (state.cursor.getTime() < dayEnd && state.cursor.getTime() < SIM_END_MS) {
      const summary = await processSlot(state.cursor);
      slots.push(summary);
      state = { ...state, cursor: new Date(summary.nextCursor) };
    }
    return { date: dateKey(dayStart), slots, nextCursor: state.cursor.toISOString() };
  });
  const summary = await getDailySummary(result.date);
  return { date: result.date, slotsRun: result.slots.length, nextCursor: result.nextCursor, summary, slots: result.slots };
}


export async function resetSimulation() {
  return withLock(async () => {
    const [p, t] = await Promise.all([Prediction.deleteMany({}), Trade.deleteMany({})]);
    await SimulationState.deleteMany({});
    await getState();
    return {
      predictionsDeleted: p.deletedCount,
      tradesDeleted: t.deletedCount,
      cursor: new Date(SIM_START_MS).toISOString(),
      note: "Database cleared. On-chain trades (if the same node is still running) are not removed; restart the Hardhat node and redeploy for a fully clean ledger.",
    };
  });
}



