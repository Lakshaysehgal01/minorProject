import mongoose from "mongoose";
import { checkChain } from "./blockchain.service";
import { getSlot } from "./dataSource.service";
import { checkMlService } from "./mlClient.service";
import { checkSync } from "./simulation.service";
import { SIM_START_MS, HOUR_MS } from "../utils/time";
import { errorMessage } from "../utils/errors";


export async function getHealth() {
  const mongo = await (async () => {
    try {
      if (mongoose.connection.readyState !== 1 || !mongoose.connection.db) return { up: false, error: "not connected" };
      await mongoose.connection.db.admin().ping();
      return { up: true };
    } catch (e) {
      return { up: false, error: errorMessage(e) };
    }
  })();


  const probeRow = getSlot(SIM_START_MS + 12 * HOUR_MS)?.[0];
  const [ml, chain] = await Promise.all([checkMlService(probeRow), checkChain()]);


  let sync: { ok: boolean; reason?: string; onChainTradeCount?: number; dbConfirmedTrades?: number } | { ok: null; reason: string } = {
    ok: null,
    reason: "skipped (mongo or chain down)",
  };
  if (mongo.up && chain.up) {
    try {
      sync = await checkSync();
    } catch (e) {
      sync = { ok: false, reason: errorMessage(e) };
    }
  }


  const healthy = mongo.up && ml.up && ml.modelsLoaded && chain.up && sync.ok === true;
  // ml.activeModel is the model that actually served the probe prediction (may be a fallback model)
  return { status: healthy ? "ok" : "degraded", components: { mongo, ml, chain, chainDbSync: sync } };
}



