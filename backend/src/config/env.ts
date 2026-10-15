import "dotenv/config";
import path from "path";
import { z } from "zod";


// "auto" = pick the highest-R2 servable model per target from the comparison CSVs at startup
// (see services/modelSelection.service.ts). lstm/ensemble get a 24-hour history payload.
const modelChoice = z.enum(["auto", "xgboost", "lightgbm", "catboost", "lstm", "ensemble"]);


// Hardhat's well-known account #0 key. Public knowledge: never use on a real network.
const HARDHAT_ACCOUNT_0_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";


const schema = z
  .object({
    PORT: z.coerce.number().int().positive().default(4000),
    MONGO_URI: z.string().min(1, "MONGO_URI is required (see .env.example)"),
    ML_SERVICE_URL: z.string().url().default("http://localhost:8000"),
    ML_MODEL: modelChoice.default("auto"),
    ML_MODEL_SOLAR: modelChoice.optional(),
    ML_MODEL_CONSUMPTION: modelChoice.optional(),
    ML_RESULTS_DIR: z.string().default("../ml-service/results"),
    DATA_CSV_PATH: z.string().default("../ml-service/data/energy_data.csv"),
    RPC_URL: z.string().url().default("http://127.0.0.1:8545"),
    OWNER_PRIVATE_KEY: z.string().default(HARDHAT_ACCOUNT_0_KEY),
    CONTRACT_ADDRESS: z.string().optional(),
    P2P_PRICE_PER_KWH: z.coerce.number().positive().default(6),
    GRID_PRICE: z.coerce.number().positive().default(8),
    FEEDIN_PRICE: z.coerce.number().positive().default(3),
    MIN_KWH: z.coerce.number().nonnegative().default(0.5),
    CORS_ORIGIN: z.string().default("http://localhost:5173"),
  })
  .refine((e) => e.GRID_PRICE > e.P2P_PRICE_PER_KWH && e.P2P_PRICE_PER_KWH > e.FEEDIN_PRICE, {
    message: "Prices must satisfy GRID_PRICE > P2P_PRICE_PER_KWH > FEEDIN_PRICE > 0",
  });


const parsed = schema.safeParse({
  ...process.env,
  // treat empty strings (e.g. `CONTRACT_ADDRESS=` in .env) as unset
  ...Object.fromEntries(Object.entries(process.env).filter(([, v]) => v === "").map(([k]) => [k, undefined])),
});


if (!parsed.success) {
  const lines = parsed.error.issues.map((i) => `  - ${i.path.join(".") || "(env)"}: ${i.message}`);
  throw new Error(`Invalid environment configuration:\n${lines.join("\n")}`);
}


const env = parsed.data;


export const config = {
  ...env,
  ML_MODEL_SOLAR: env.ML_MODEL_SOLAR ?? env.ML_MODEL,
  ML_MODEL_CONSUMPTION: env.ML_MODEL_CONSUMPTION ?? env.ML_MODEL,
  ML_RESULTS_DIR: path.resolve(process.cwd(), env.ML_RESULTS_DIR),
  DATA_CSV_PATH: path.resolve(process.cwd(), env.DATA_CSV_PATH),
  CORS_ORIGINS: env.CORS_ORIGIN.split(",").map((s) => s.trim()),
};


export type Config = typeof config;



