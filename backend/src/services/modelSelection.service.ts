import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import { config } from "../config/env";


export type ModelName = "xgboost" | "lightgbm" | "catboost" | "lstm" | "ensemble";
export type Target = "solar" | "consumption";


export interface Selection {
  model: ModelName;
  r2: number | null; // test-period R2 from the comparison table (null when pinned and not listed)
  source: "auto" | "pinned";
}


// Comparison-table row name -> the ?model= value the ML service understands. Rows with other
// names (Prophet, LinearRegression, RandomForest, ...) cannot be served by name and are ignored.
const SERVABLE: Record<string, ModelName> = {
  XGBRegressor: "xgboost",
  LGBMRegressor: "lightgbm",
  CatBoostRegressor: "catboost",
  LSTM: "lstm",
  Ensemble: "ensemble",
};


// lstm/ensemble take a 24-hour history payload instead of a single row.
export const isSequenceModel = (m: string): boolean => m === "lstm" || m === "ensemble";


interface Ranked {
  model: ModelName;
  r2: number;
}


function rankedFromCsv(target: Target): Ranked[] {
  const file = `${target}_model_comparison.csv`;
  const full = path.join(config.ML_RESULTS_DIR, file);
  if (!fs.existsSync(full)) {
    throw new Error(
      `ML_MODEL=auto needs ${full} to choose the best ${target} model. ` +
        "Run `python results/model_comparison.py` in ml-service, or set ML_MODEL / ML_MODEL_SOLAR / ML_MODEL_CONSUMPTION to a specific model."
    );
  }
  const rows = parse(fs.readFileSync(full, "utf8"), { columns: true, skip_empty_lines: true }) as Record<string, string>[];
  const ranked: Ranked[] = [];
  for (const r of rows) {
    const model = SERVABLE[r.Model];
    const r2 = Number(r.R2);
    if (model && Number.isFinite(r2)) ranked.push({ model, r2 });
  }
  if (!ranked.length) {
    throw new Error(`${file} lists no servable model (expected one of ${Object.keys(SERVABLE).join(", ")}).`);
  }
  return ranked.sort((a, b) => b.r2 - a.r2);
}


let selected: Record<Target, Selection> | null = null;
// Full ranked list per target (used for fallback). For pinned models, this is a single-entry list.
let fallbackOrder: Record<Target, ModelName[]> | null = null;


function pick(target: Target, configured: ModelName | "auto"): { selection: Selection; order: ModelName[] } {
  if (configured !== "auto") {
    return { selection: { model: configured, r2: null, source: "pinned" }, order: [configured] };
  }
  const ranked = rankedFromCsv(target);
  const best = ranked[0];
  console.log(`[models] ${target}: auto-selected ${best.model} (highest test R2 among servable models: ${best.r2.toFixed(4)})`);
  // Deduplicate in case the same model appears twice (shouldn't happen, but guard it)
  const seen = new Set<ModelName>();
  const order: ModelName[] = [];
  for (const r of ranked) {
    if (!seen.has(r.model)) { seen.add(r.model); order.push(r.model); }
  }
  return { selection: { model: best.model, r2: best.r2, source: "auto" }, order };
}


// Call once at startup, before any prediction. Throws if `auto` cannot be resolved.
export function resolveModels(): void {
  const solar = pick("solar", config.ML_MODEL_SOLAR);
  const consumption = pick("consumption", config.ML_MODEL_CONSUMPTION);
  selected = { solar: solar.selection, consumption: consumption.selection };
  fallbackOrder = { solar: solar.order, consumption: consumption.order };
}


export function getSelection(target: Target): Selection {
  if (!selected) throw new Error("Models not resolved yet: resolveModels() must run at startup.");
  return selected[target];
}


/** Ordered list of models to try for a target, best-first. Falls back down if a model fails. */
export function getFallbackOrder(target: Target): ModelName[] {
  if (!fallbackOrder) throw new Error("Models not resolved yet: resolveModels() must run at startup.");
  return fallbackOrder[target];
}


export function getActiveModels() {
  return { solar: getSelection("solar"), consumption: getSelection("consumption") };
}



