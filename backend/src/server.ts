import { createApp } from "./app";
import { connectDb } from "./config/db";
import { config } from "./config/env";
import { loadData } from "./services/dataSource.service";
import { getActiveModels, resolveModels } from "./services/modelSelection.service";

const dns = require("node:dns");

dns.setServers(["8.8.8.8", "1.1.1.1"]);

async function main() {
  resolveModels();


  console.log("Loading dataset...");
  const t0 = Date.now();
  await loadData();
  console.log(`Dataset loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);


  await connectDb();
  console.log("MongoDB connected");


  createApp().listen(config.PORT, () => {
    console.log(`Backend listening on http://localhost:${config.PORT}`);
    const m = getActiveModels();
    console.log(`ML: ${config.ML_SERVICE_URL} (solar=${m.solar.model} [${m.solar.source}], consumption=${m.consumption.model} [${m.consumption.source}])`);
    console.log(`Chain: ${config.RPC_URL}`);
  });
}


main().catch((err) => {
  console.error("Startup failed:", err instanceof Error ? err.message : err);
  process.exit(1);
});



