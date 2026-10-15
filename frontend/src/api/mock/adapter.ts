import { AxiosError, CanceledError, type AxiosAdapter, type AxiosResponse, type InternalAxiosRequestConfig } from "axios";
import health from "./health.json";
import results from "./results.json";
import {
  DATASET_START,
  MOCK_MODELS,
  N_HOUSEHOLDS,
  SIM_END,
  allTrades,
  dailySummary,
  forecast,
  getCursor,
  households,
  predictionHours,
  setCursor,
  slotSummary,
  stats,
} from "./data";


const HOUR = 3_600_000;
const DAY = 24 * HOUR;


interface Reply {
  status: number;
  data: unknown;
}


const ok = (data: unknown): Reply => ({ status: 200, data });
const fail = (status: number, error: string, extra: object = {}): Reply => ({ status, data: { error, ...extra } });


function parseDate(s: unknown): number | null {
  if (typeof s !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(s)) return null;
  const ms = Date.parse(`${s}T00:00:00Z`);
  return Number.isNaN(ms) || new Date(ms).toISOString().slice(0, 10) !== s ? null : ms;
}


const badDate = () =>
  fail(400, "Validation failed", { issues: [{ path: "date", message: "must be a valid date in YYYY-MM-DD format" }] });


function sleep(ms: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new CanceledError());
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new CanceledError());
    });
  });
}


let busy = false;


const COMPLETE_MSG =
  "Simulation complete: the whole test period (to 2026-12-31 23:00) has been replayed. POST /simulate/reset to start over.";


async function route(method: string, path: string, params: Record<string, unknown>, signal?: AbortSignal | null): Promise<Reply> {
  if (method === "get" && path === "/health") {
    if (import.meta.env.VITE_MOCK_HEALTH === "degraded") {
      return {
        status: 503,
        data: {
          ...health,
          status: "degraded",
          components: { ...health.components, ml: { up: false, modelsLoaded: false, error: "ML service unreachable at http://localhost:8000" } },
        },
      };
    }
    return ok(health);
  }
  if (method === "get" && path === "/stats") return ok(stats());
  if (method === "get" && path === "/models/active") return ok(MOCK_MODELS);
  if (method === "get" && path === "/households") return ok(households());


  if (method === "get" && path === "/results/models") {
    if (import.meta.env.VITE_MOCK_NO_RESULTS === "true") {
      return fail(404, "Model comparison file(s) not found in ../ml-service/results: solar_model_comparison.csv. Generate them with `python results/model_comparison.py` in ml-service.");
    }
    return ok(results);
  }


  if (method === "get" && path === "/daily-summary") {
    const day = parseDate(params.date);
    if (day === null) return badDate();
    const summary = dailySummary(day);
    return summary ? ok(summary) : fail(404, `No simulated data for ${params.date}. Run the simulation for that day first.`);
  }


  if (method === "get" && path === "/trades") {
    const page = Number(params.page ?? 1);
    const limit = Number(params.limit ?? 50);
    let list = allTrades();
    if (params.date !== undefined) {
      const day = parseDate(params.date);
      if (day === null) return badDate();
      list = list.filter((t) => Date.parse(t.slotTimestamp) >= day && Date.parse(t.slotTimestamp) < day + DAY);
    }
    const newestFirst = [...list].reverse();
    return ok({ page, limit, total: newestFirst.length, trades: newestFirst.slice((page - 1) * limit, page * limit) });
  }


  const hhTrades = /^\/trades\/household\/(\d+)$/.exec(path);
  if (method === "get" && hhTrades) {
    const id = Number(hhTrades[1]);
    if (id >= N_HOUSEHOLDS) return fail(404, `Unknown household ${id}`);
    const page = Number(params.page ?? 1);
    const limit = Number(params.limit ?? 50);
    const mine = [...allTrades()].reverse().filter((t) => t.sellerId === id || t.buyerId === id);
    return ok({ householdId: id, page, limit, total: mine.length, trades: mine.slice((page - 1) * limit, page * limit) });
  }


  const pred = /^\/predictions\/(\d+)$/.exec(path);
  if (method === "get" && pred) {
    const id = Number(pred[1]);
    if (id >= N_HOUSEHOLDS) return fail(404, `Unknown household ${id}`);
    const day = params.date === undefined ? Math.floor((getCursor() - HOUR) / DAY) * DAY : parseDate(params.date);
    if (day === null) return badDate();
    const hours = predictionHours(id, day);
    return hours.length
      ? ok({ householdId: id, date: new Date(day).toISOString().slice(0, 10), hours })
      : fail(404, `No predictions for household ${id} on ${new Date(day).toISOString().slice(0, 10)}`);
  }


  const fc = /^\/forecast\/(\d+)$/.exec(path);
  if (method === "get" && fc) {
    const id = Number(fc[1]);
    const day = parseDate(params.date);
    if (day === null) return badDate();
    if (id >= N_HOUSEHOLDS) return fail(404, `Unknown household ${id}`);
    if (day < DATASET_START || day >= SIM_END) {
      return fail(404, `Dataset has no weather data for ${params.date} (covers 2026-01-01 to 2026-12-31)`);
    }
    await sleep(600, signal); // the real endpoint calls the ML service
    return ok(forecast(id, day));
  }


  if (method === "post" && path === "/simulate/tick") {
    if (busy) return fail(409, "Simulation busy: another run is in progress.");
    if (getCursor() >= SIM_END) return fail(409, COMPLETE_MSG);
    busy = true;
    try {
      await sleep(500, signal);
      const slot = getCursor();
      setCursor(slot + HOUR);
      return ok(slotSummary(slot));
    } finally {
      busy = false;
    }
  }


  if (method === "post" && path === "/simulate/day") {
    if (busy) return fail(409, "Simulation busy: another run is in progress.");
    if (getCursor() >= SIM_END) return fail(409, COMPLETE_MSG);
    busy = true;
    try {
      await sleep(3000, signal);
      const dayStart = Math.floor(getCursor() / DAY) * DAY;
      const dayEnd = Math.min(dayStart + DAY, SIM_END);
      const slots = [];
      for (let s = getCursor(); s < dayEnd; s += HOUR) slots.push(slotSummary(s));
      setCursor(dayEnd);
      return ok({
        date: new Date(dayStart).toISOString().slice(0, 10),
        slotsRun: slots.length,
        nextCursor: new Date(dayEnd).toISOString(),
        summary: dailySummary(dayStart),
        slots,
      });
    } finally {
      busy = false;
    }
  }


  return fail(404, "Not found");
}


export const mockAdapter: AxiosAdapter = async (config: InternalAxiosRequestConfig) => {
  const method = (config.method ?? "get").toLowerCase();
  const path = (config.url ?? "").replace(/^https?:\/\/[^/]+/, "");
  await sleep(120, config.signal as AbortSignal | undefined);
  const reply = await route(method, path, (config.params ?? {}) as Record<string, unknown>, config.signal as AbortSignal | undefined);


  const response: AxiosResponse = { data: reply.data, status: reply.status, statusText: "", headers: {}, config, request: {} };
  const valid = config.validateStatus ? config.validateStatus(reply.status) : reply.status >= 200 && reply.status < 300;
  if (!valid) {
    throw new AxiosError(`Request failed with status code ${reply.status}`, AxiosError.ERR_BAD_REQUEST, config, {}, response);
  }
  return response;
};





