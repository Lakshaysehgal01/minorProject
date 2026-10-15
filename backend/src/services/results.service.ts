import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import { config } from "../config/env";
import { httpError } from "../utils/errors";


type Row = Record<string, string | number>;


function readTable(file: string) {
  const full = path.join(config.ML_RESULTS_DIR, file);
  if (!fs.existsSync(full)) return null;
  const records = parse(fs.readFileSync(full, "utf8"), { columns: true, skip_empty_lines: true }) as Record<string, string>[];
  const rows: Row[] = records.map((r) =>
    Object.fromEntries(Object.entries(r).map(([k, v]) => [k, k === "Model" || v === "" || Number.isNaN(Number(v)) ? v : Number(v)]))
  );
  return { file, generatedAt: fs.statSync(full).mtime.toISOString(), rows };
}


export function getModelComparison() {
  const solar = readTable("solar_model_comparison.csv");
  const consumption = readTable("consumption_model_comparison.csv");
  const missing = [
    !solar && "solar_model_comparison.csv",
    !consumption && "consumption_model_comparison.csv",
  ].filter(Boolean);
  if (missing.length) {
    throw httpError(
      404,
      `Model comparison file(s) not found in ${config.ML_RESULTS_DIR}: ${missing.join(", ")}. Generate them with \`python results/model_comparison.py\` in ml-service.`
    );
  }
  return { solar, consumption };
}



