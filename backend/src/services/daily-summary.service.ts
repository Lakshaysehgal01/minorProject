import { config } from "../config/env";
import { Prediction, PredictionDoc } from "../models/prediction.model";
import { Trade, TradeDoc } from "../models/trade.model";
import { computeMae } from "../utils/metrics";
import { DAY_MS, parseDateKey } from "../utils/time";
import { httpError } from "../utils/errors";
import { computeSavings } from "./savings.service";


type Pred = Pick<
  PredictionDoc,
  | "householdId" | "slotTimestamp" | "predictedSolarKwh" | "predictedConsumptionKwh" | "actualSolarKwh"
  | "actualConsumptionKwh" | "surplusKwh" | "isSolarAnomaly" | "isConsumptionAnomaly"
>;
type ConfirmedTrade = Pick<TradeDoc, "slotTimestamp" | "sellerId" | "buyerId" | "kWh">;


const pct = (num: number, den: number): number | null => (den > 0 ? (num / den) * 100 : null);


// Shared by the daily summary and /stats so both use identical definitions.
export function buildSummary(predictions: Pred[], confirmedTrades: ConfirmedTrade[], failedTradeCount: number) {
  let predSolar = 0, actSolar = 0, predCons = 0, actCons = 0;
  let predictedDeficitKwh = 0, actualDeficitKwh = 0;
  const slots = new Set<number>();
  for (const p of predictions) {
    slots.add(p.slotTimestamp.getTime());
    predSolar += p.predictedSolarKwh;
    actSolar += p.actualSolarKwh;
    predCons += p.predictedConsumptionKwh;
    actCons += p.actualConsumptionKwh;
    // predicted deficit counts only households eligible as buyers (deficit > MIN_KWH)
    if (-p.surplusKwh > config.MIN_KWH) predictedDeficitKwh += -p.surplusKwh;
    actualDeficitKwh += Math.max(0, p.actualConsumptionKwh - p.actualSolarKwh);
  }


  const savings = computeSavings(predictions, confirmedTrades);
  const mae = computeMae(predictions);


  return {
    slotsCovered: slots.size,
    solar: { predictedKwh: predSolar, actualKwh: actSolar },
    consumption: { predictedKwh: predCons, actualKwh: actCons },
    trading: {
      tradedKwh: savings.tradedKwh,
      tradeCount: confirmedTrades.length,
      failedTradeCount,
      predictedDeficitKwh,
      actualDeficitKwh,
      percentDeficitMetP2P: pct(savings.tradedKwh, predictedDeficitKwh),
      percentActualDeficitMetP2P: pct(savings.tradedKwh, actualDeficitKwh),
    },
    savings,
    predictionError: mae,
  };
}


export async function getDailySummary(date: string) {
  const dayStart = parseDateKey(date);
  if (dayStart === null) throw httpError(400, "date must be a valid YYYY-MM-DD");
  const range = { $gte: new Date(dayStart), $lt: new Date(dayStart + DAY_MS) };


  const [predictions, trades, failedTradeCount] = await Promise.all([
    Prediction.find({ slotTimestamp: range }).lean(),
    Trade.find({ slotTimestamp: range, status: "confirmed" }).lean(),
    Trade.countDocuments({ slotTimestamp: range, status: "failed" }),
  ]);
  if (!predictions.length) throw httpError(404, `No simulated data for ${date}. Run the simulation for that day first.`);


  const summary = buildSummary(predictions, trades, failedTradeCount);
  return { date, complete: summary.slotsCovered === 24, ...summary };
}



