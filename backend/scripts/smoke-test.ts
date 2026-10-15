// End-to-end smoke test against a RUNNING backend (npm run dev) + ML service + Hardhat node + Mongo.
// Runs one simulated day, prints trades / balances / savings, and asserts ledger invariants.
import "dotenv/config";


const BASE = process.env.BACKEND_URL ?? `http://localhost:${process.env.PORT ?? 4000}`;


let failures = 0;
function check(ok: boolean, label: string, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures++;
}


async function api<T = any>(path: string, init?: RequestInit): Promise<{ status: number; body: T }> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: AbortSignal.timeout(10 * 60_000) });
  } catch (e) {
    console.error(`Cannot reach backend at ${BASE}. Start it with \`npm run dev\` (from the repo root). (${(e as Error).message})`);
    process.exit(1);
  }
  return { status: res.status, body: (await res.json().catch(() => ({}))) as T };
}


const fmt = (n: number | null | undefined, d = 2) => (n === null || n === undefined ? "n/a" : n.toFixed(d));


async function main() {
  console.log(`Smoke test against ${BASE}\n`);


  console.log("1. Health");
  const health = await api("/health");
  console.log(JSON.stringify(health.body.components, null, 2));
  if (health.status !== 200) {
    console.error("\nA dependency is not healthy (see above). Run order: Hardhat node -> deploy -> ML service -> Mongo -> backend.");
    process.exit(1);
  }


  console.log("\n2. Simulating one day (POST /simulate/day)...");
  const t0 = Date.now();
  const dayRes = await api("/simulate/day", { method: "POST" });
  if (dayRes.status !== 200) {
    console.error(`Simulation failed (${dayRes.status}): ${JSON.stringify(dayRes.body)}`);
    process.exit(1);
  }
  const day = dayRes.body;
  console.log(`   ${day.slotsRun} slots for ${day.date} in ${((Date.now() - t0) / 1000).toFixed(1)}s, next cursor ${day.nextCursor}`);


  console.log(`\n3. Trades on ${day.date} (newest first, first 15)`);
  const tradesRes = await api(`/trades?date=${day.date}&limit=15`);
  console.log("   slot                      seller -> buyer   kWh     credits  status     tradeId  tx");
  for (const t of tradesRes.body.trades) {
    console.log(
      `   ${new Date(t.slotTimestamp).toISOString()}  ${String(t.sellerId).padStart(4)} -> ${String(t.buyerId).padEnd(4)}  ${t.kWh.toFixed(3).padStart(7)} ${(t.costMilli / 1000).toFixed(3).padStart(8)}  ${t.status.padEnd(9)}  ${String(t.tradeId ?? "-").padStart(5)}  ${(t.txHash ?? "").slice(0, 12)}`
    );
  }
  console.log(`   total trades that day: ${tradesRes.body.total}`);


  console.log("\n4. On-chain balances (kWh / credits)");
  const hh = (await api("/households")).body as any[];
  for (const h of hh) {
    console.log(
      `   #${String(h.householdId).padStart(2)} ${h.tier.padEnd(6)} ${String(h.panelCapacityKw).padStart(5)} kW   energy ${h.energyBalanceKwh.toFixed(3).padStart(9)}   credits ${h.tokenBalance.toFixed(3).padStart(9)}`
    );
  }


  console.log("\n5. Day summary");
  const s = day.summary;
  console.log(`   solar   predicted ${fmt(s.solar.predictedKwh)} kWh   actual ${fmt(s.solar.actualKwh)} kWh`);
  console.log(`   usage   predicted ${fmt(s.consumption.predictedKwh)} kWh   actual ${fmt(s.consumption.actualKwh)} kWh`);
  console.log(`   traded ${fmt(s.trading.tradedKwh)} kWh in ${s.trading.tradeCount} trades (${s.trading.failedTradeCount} failed)`);
  console.log(`   % of deficit demand met P2P: ${fmt(s.trading.percentDeficitMetP2P, 1)}% (vs predicted deficit), ${fmt(s.trading.percentActualDeficitMetP2P, 1)}% (vs actual deficit)`);
  console.log(`   savings  gross ${fmt(s.savings.grossSavings)}   realized ${fmt(s.savings.realizedSavings)}   perfect-foresight bound ${fmt(s.savings.perfectForesightSavings)}   (undelivered ${fmt(s.savings.undeliveredKwh)} kWh)`);
  console.log(`   grid-only baseline cost ${fmt(s.savings.gridOnlyBaselineCost)}   P2P cost ${fmt(s.savings.p2pCost)}`);
  const mae = s.predictionError;
  console.log(`   MAE solar ${fmt(mae.solar.mae, 3)} (excl. anomalies ${fmt(mae.solar.maeExcludingAnomalies, 3)}, ${mae.solar.nAnomalies} anomaly hrs)`);
  console.log(`   MAE usage ${fmt(mae.consumption.mae, 3)} (excl. anomalies ${fmt(mae.consumption.maeExcludingAnomalies, 3)}, ${mae.consumption.nAnomalies} anomaly hrs)`);


  console.log("\n6. Invariants");
  check(day.slotsRun >= 1, "simulated at least one slot");
  check(s.trading.tradeCount > 0, "at least one trade settled on-chain");
  check(s.trading.failedTradeCount === 0, "no failed trades", `${s.trading.failedTradeCount} failed`);


  const energySum = hh.reduce((a, h) => a + h.energyBalanceMilli, 0);
  const tokenSum = hh.reduce((a, h) => a + h.tokenBalanceMilli, 0);
  check(energySum === 0, "on-chain energy balances sum to 0", `sum=${energySum}`);
  check(tokenSum === 0, "on-chain token balances sum to 0", `sum=${tokenSum}`);


  // Rebuild expected balances from every confirmed trade in Mongo and compare with the chain.
  const expected = new Map<number, { energy: number; token: number }>(hh.map((h) => [h.householdId, { energy: 0, token: 0 }]));
  let confirmed = 0;
  for (let page = 1; ; page++) {
    const r = (await api(`/trades?status=confirmed&limit=500&page=${page}`)).body;
    for (const t of r.trades) {
      confirmed++;
      expected.get(t.sellerId)!.energy -= t.kWhMilli;
      expected.get(t.buyerId)!.energy += t.kWhMilli;
      expected.get(t.buyerId)!.token -= t.costMilli;
      expected.get(t.sellerId)!.token += t.costMilli;
    }
    if (page * 500 >= r.total) break;
  }
  const stats = (await api("/stats")).body;
  check(stats.onChainTradeCount === confirmed, "on-chain trade count equals Mongo confirmed trades", `chain=${stats.onChainTradeCount} db=${confirmed}`);
  const mismatches = hh.filter((h) => {
    const e = expected.get(h.householdId)!;
    return e.energy !== h.energyBalanceMilli || e.token !== h.tokenBalanceMilli;
  });
  check(mismatches.length === 0, "on-chain balances equal Mongo-derived balances for all 25 households", mismatches.length ? `mismatch: ${mismatches.map((m) => m.householdId).join(",")}` : "");


  console.log(`\n7. Overall stats: ${stats.simulation.slotsCompleted}/${stats.simulation.totalSlots} slots, ${fmt(stats.trading.tradedKwh)} kWh traded, gross savings ${fmt(stats.savings.grossSavings)}`);


  console.log(failures ? `\nSMOKE TEST FAILED (${failures} check(s))` : "\nSMOKE TEST PASSED");
  process.exit(failures ? 1 : 0);
}


main().catch((e) => {
  console.error(e);
  process.exit(1);
});



