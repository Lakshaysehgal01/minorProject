#!/usr/bin/env node
// Starts the whole stack in order: Hardhat node -> deploy -> ML service -> backend -> frontend.
// Usage: npm run start:all [-- --reset] [-- --mock]
//   --reset  call POST /simulate/reset once the backend is up (needed after a fresh chain if
//            MongoDB still holds trades from an earlier run)
//   --mock   skip the backend/ML/chain and run only the frontend in mock mode
// Ctrl+C stops everything.
const { spawn, spawnSync } = require("child_process");
const net = require("net");
const path = require("path");
const fs = require("fs");


const ROOT = path.resolve(__dirname, "..");
const IS_WIN = process.platform === "win32";
const args = new Set(process.argv.slice(2));


const PORTS = { chain: 8545, ml: 8000, backend: 4000, frontend: 5173 };
const COLORS = { chain: 36, deploy: 33, ml: 35, backend: 32, frontend: 34, launcher: 37 };
const children = [];
let shuttingDown = false;


function log(tag, msg) {
  const c = COLORS[tag] ?? 37;
  for (const line of String(msg).split(/\r?\n/)) {
    if (line.trim()) console.log(`\x1b[${c}m[${tag}]\x1b[0m ${line}`);
  }
}


function probe(port, host) {
  return new Promise((resolve) => {
    const s = net.connect({ port, host });
    s.once("connect", () => (s.destroy(), resolve(true)));
    s.once("error", () => resolve(false));
  });
}


// "localhost" can resolve to IPv4 or IPv6 depending on the machine, so a dev server may be
// listening on either; accept both.
async function portInUse(port) {
  return (await probe(port, "127.0.0.1")) || (await probe(port, "::1"));
}


function start(tag, command, argv, cwd) {
  const child = spawn(command, argv, { cwd, shell: IS_WIN, detached: !IS_WIN, env: process.env });
  children.push(child);
  child.stdout.on("data", (d) => log(tag, d));
  child.stderr.on("data", (d) => log(tag, d));
  child.on("exit", (code) => {
    if (!shuttingDown) log(tag, `exited with code ${code}`);
  });
  return child;
}


function runOnce(tag, command, argv, cwd) {
  return new Promise((resolve, reject) => {
    const child = start(tag, command, argv, cwd);
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${tag} failed (exit ${code})`))));
  });
}


// Waits until check() is true. If `child` dies first, fails immediately with its exit code
// instead of waiting out the timeout. Logs a progress line every 15 s.
async function waitFor(name, check, timeoutMs = 180_000, child = null) {
  const t0 = Date.now();
  let lastNote = t0;
  while (Date.now() - t0 < timeoutMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`${name} exited with code ${child.exitCode} before it became ready (see its output above)`);
    }
    try {
      if (await check()) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() - lastNote >= 15_000) {
      lastNote = Date.now();
      log("launcher", `still waiting for ${name} (${Math.round((Date.now() - t0) / 1000)}s of ${timeoutMs / 1000}s)...`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`${name} did not come up within ${timeoutMs / 1000}s (see its output above)`);
}


function killTree(child) {
  if (!child.pid || child.exitCode !== null) return;
  if (IS_WIN) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
}


function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  log("launcher", "stopping all services...");
  children.forEach(killTree);
  setTimeout(() => process.exit(code), 500);
}
process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));


async function main() {
  const mock = args.has("--mock");
  const needed = mock ? ["frontend"] : Object.keys(PORTS);
  const busy = [];
  for (const k of needed) if (await portInUse(PORTS[k])) busy.push(`${k} (port ${PORTS[k]})`);
  if (busy.length) {
    throw new Error(
      `Already in use: ${busy.join(", ")}. Stop the existing processes first (e.g. close their terminals).`
    );
  }


  if (mock) {
    log("launcher", "mock mode: starting only the frontend");
    const fe = start("frontend", "npm", ["--prefix", "frontend", "run", "dev:mock"], ROOT);
    await waitFor("frontend", () => portInUse(PORTS.frontend), 180_000, fe);
    log("launcher", "ready: http://localhost:5173 (mock data). Ctrl+C to stop.");
    return;
  }


  const backendEnv = path.join(ROOT, "backend", ".env");
  if (!fs.existsSync(backendEnv)) {
    throw new Error("backend/.env is missing. Copy backend/.env.example to backend/.env and set MONGO_URI.");
  }
  const python = path.join(
    ROOT,
    "ml-service",
    "venv",
    IS_WIN ? "Scripts" : "bin",
    IS_WIN ? "python.exe" : "python"
  );
  if (!fs.existsSync(python)) {
    throw new Error(`ML virtualenv not found at ${python}. See README section 11 to create it.`);
  }


  log("launcher", "1/5 starting Hardhat node...");
  const chain = start("chain", "npm", ["--prefix", "contracts", "run", "chain"], ROOT);
  await waitFor(
    "chain",
    async () => {
      const r = await fetch("http://127.0.0.1:8545", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", method: "eth_chainId", params: [], id: 1 }),
      });
      return r.ok;
    },
    180_000,
    chain
  );


  log("launcher", "2/5 deploying contract and registering 25 households...");
  await runOnce("deploy", "npm", ["--prefix", "contracts", "run", "deploy"], ROOT);


  log("launcher", "3/5 starting ML service...");
  const ml = start("ml", `"${python}"`, ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], path.join(ROOT, "ml-service"));
  // Loading every trained model (and importing TensorFlow) can take several minutes.
  await waitFor("ML service", async () => (await fetch("http://127.0.0.1:8000/health")).ok, 600_000, ml);


  log("launcher", "4/5 starting backend (loads the dataset, then connects to MongoDB)...");
  const backend = start("backend", "npm", ["--prefix", "backend", "run", "dev"], ROOT);
  backend.on("exit", (code) => {
    if (!shuttingDown && code) shutdown(1);
  });
  await waitFor(
    "backend",
    async () => {
      await fetch("http://localhost:4000/health"); // any HTTP status means it is listening
      return true;
    },
    300_000,
    backend
  );


  if (args.has("--reset")) {
    log("launcher", "resetting simulation state (--reset)...");
    const r = await fetch("http://localhost:4000/simulate/reset", { method: "POST" });
    log("launcher", `reset: HTTP ${r.status}`);
  } else {
    const h = await (await fetch("http://localhost:4000/health")).json();
    if (h.components?.chainDbSync?.ok === false) {
      log(
        "launcher",
        "WARNING: MongoDB holds trades from an earlier run but this chain is fresh. " +
          "Restart with `npm run start:all -- --reset` (clears MongoDB simulation data) before running a day."
      );
    }
  }


  log("launcher", "5/5 starting frontend...");
  const frontend = start("frontend", "npm", ["--prefix", "frontend", "run", "dev"], ROOT);
  await waitFor("frontend", () => portInUse(PORTS.frontend), 180_000, frontend);


  log("launcher", "all services are up. Open http://localhost:5173 and click 'Run Next Day'. Ctrl+C stops everything.");
}


main().catch((e) => {
  log("launcher", `ERROR: ${e.message}`);
  shutdown(1);
});



