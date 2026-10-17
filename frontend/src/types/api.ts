// Response shapes confirmed against backend/src/controllers and services.
// All ISO timestamps are naive local time stored as UTC: never convert to the browser's zone.


export type Tier = "small" | "medium" | "large";
export type TradeStatus = "pending" | "confirmed" | "failed";


export interface MaeResult {
  mae: number | null;
  maeExcludingAnomalies: number | null;
  n: number;
  nAnomalies: number;
}


export interface SavingsBlock {
  tradedKwh: number;
  buyerSavings: number;
  sellerGains: number;
  grossSavings: number;
  gridOnlyBaselineCost: number;
  p2pCost: number;
  undeliveredKwh: number;
  realizedSavings: number;
  perfectForesightTradedKwh: number;
  perfectForesightSavings: number;
}


export interface SummaryBlocks {
  slotsCovered: number;
  solar: { predictedKwh: number; actualKwh: number };
  consumption: { predictedKwh: number; actualKwh: number };
  trading: {
    tradedKwh: number;
    tradeCount: number;
    failedTradeCount: number;
    predictedDeficitKwh: number;
    actualDeficitKwh: number;
    percentDeficitMetP2P: number | null;
    percentActualDeficitMetP2P: number | null;
  };
  savings: SavingsBlock;
  predictionError: { solar: MaeResult; consumption: MaeResult };
}


export interface DailySummary extends SummaryBlocks {
  date: string;
  complete: boolean;
}


export interface SlotSummary {
  slot: string;
  tradesConfirmed: number;
  tradesFailed: number;
  kWhTraded: number;
  predictedSurplusKwh: number;
  predictedDeficitKwh: number;
  nextCursor: string;
}


export interface RunDayResponse {
  date: string;
  slotsRun: number;
  nextCursor: string;
  summary: DailySummary;
  slots: SlotSummary[];
}


export interface Household {
  householdId: number;
  tier: Tier;
  panelCapacityKw: number;
  energyBalanceMilli: number;
  tokenBalanceMilli: number;
  energyBalanceKwh: number;
  tokenBalance: number;
}


export interface Trade {
  _id: string;
  tradeId?: number;
  sellerId: number;
  buyerId: number;
  kWh: number;
  kWhMilli: number;
  pricePerKwh: number;
  costMilli: number;
  txHash?: string;
  status: TradeStatus;
  error?: string;
  slotTimestamp: string;
  createdAt: string;
}


export interface TradesPage {
  householdId?: number;
  page: number;
  limit: number;
  total: number;
  trades: Trade[];
}


export interface PredictionHour {
  slotTimestamp: string;
  householdId: number;
  predictedSolarKwh: number;
  predictedConsumptionKwh: number;
  actualSolarKwh: number;
  actualConsumptionKwh: number;
  surplusKwh: number;
  modelUsed: string;
  isSolarAnomaly: boolean;
  isConsumptionAnomaly: boolean;
}


export interface PredictionsResponse {
  householdId: number;
  date: string;
  hours: PredictionHour[];
}


export interface ForecastHour {
  slotTimestamp: string;
  hour: number;
  predictedSolarKwh: number;
  predictedConsumptionKwh: number;
  predictedSurplusKwh: number;
  weatherInput: { temperature: number; cloudCover: number; humidity: number; windSpeed: number };
}


export type ModelName = "xgboost" | "lightgbm" | "catboost" | "lstm" | "ensemble";


export interface ActiveModel {
  model: ModelName;
  r2: number | null; // test-period R2 the choice was based on (null when pinned)
  source: "auto" | "pinned";
}


export interface ActiveModels {
  solar: ActiveModel;
  consumption: ActiveModel;
}


export interface ForecastResponse {
  householdId: number;
  date: string;
  modelUsed: string;
  models: ActiveModels;
  inSample: boolean;
  note: string;
  hours: ForecastHour[];
}


export interface SimulationStatus {
  cursor: string;
  slotsCompleted: number;
  totalSlots: number;
  complete: boolean;
  contractAddress: string | null;
  chainId: number | null;
}


export interface Stats extends SummaryBlocks {
  simulation: SimulationStatus;
  onChainTradeCount: number | null;
}


export interface ModelRow {
  Model: string;
  RMSE: number;
  MSE: number;
  MAE: number;
  R2: number;
}


export interface ModelTable {
  file: string;
  generatedAt: string;
  rows: ModelRow[];
}


export interface ModelResults {
  solar: ModelTable;
  consumption: ModelTable;
}


export interface HealthResponse {
  status: "ok" | "degraded";
  components: {
    mongo: { up: boolean; error?: string };
    ml: { up: boolean; modelsLoaded: boolean; activeModel?: string; error?: string };
    chain: { up: boolean; blockNumber?: number; tradeCount?: number; error?: string };
    chainDbSync: { ok: boolean | null; reason?: string; onChainTradeCount?: number; dbConfirmedTrades?: number };
  };
}



