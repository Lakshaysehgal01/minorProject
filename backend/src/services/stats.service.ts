import { Prediction } from "../models/prediction.model";
import { Trade } from "../models/trade.model";
import { getOnChainTradeCount } from "./blockchain.service";
import { buildSummary } from "./daily-summary.service";
import { getSimulationStatus } from "./simulation.service";


export async function getStats() {
  const [predictions, trades, failedTradeCount, status] = await Promise.all([
    Prediction.find({}).lean(),
    Trade.find({ status: "confirmed" }).lean(),
    Trade.countDocuments({ status: "failed" }),
    getSimulationStatus(),
  ]);
  const onChainTradeCount = await getOnChainTradeCount().catch(() => null);
  return {
    simulation: status,
    onChainTradeCount,
    ...buildSummary(predictions, trades, failedTradeCount),
  };
}



