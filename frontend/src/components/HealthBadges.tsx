import type { Resource } from "../hooks/useResource";
import type { HealthResponse } from "../types/api";
import { Badge, type Tone } from "./Badge";


interface Pill {
  name: string;
  tone: Tone;
  state: string;
  detail: string;
}


function buildPills(health: Resource<HealthResponse>): Pill[] {
  if (health.loading) {
    return ["Backend", "ML service", "Chain"].map((name) => ({ name, tone: "slate", state: "checking", detail: "Checking…" }));
  }
  if (health.error || !health.data) {
    const why = health.error?.message ?? "No response";
    return [
      { name: "Backend", tone: "red", state: "offline", detail: why },
      { name: "ML service", tone: "slate", state: "unknown", detail: "Backend is offline" },
      { name: "Chain", tone: "slate", state: "unknown", detail: "Backend is offline" },
    ];
  }
  const { mongo, ml, chain, chainDbSync } = health.data.components;
  const mongoNote = `MongoDB: ${mongo.up ? "up" : `down (${mongo.error ?? "unknown"})`}`;
  const syncNote =
    chainDbSync.ok === false ? ` Chain/DB out of sync: ${chainDbSync.reason ?? ""}` : "";
  const mlState = (() => {
    if (!ml.up) return "down";
    if (!ml.modelsLoaded) return "no models";
    return ml.activeModel ? `ok (${ml.activeModel})` : "ok";
  })();
  return [
    { name: "Backend", tone: mongo.up ? "green" : "amber", state: mongo.up ? "online" : "degraded", detail: mongoNote },
    {
      name: "ML service",
      tone: ml.up && ml.modelsLoaded ? "green" : "red",
      state: mlState,
      detail: ml.error ?? (ml.activeModel ? `Serving predictions with: ${ml.activeModel}` : "Model probe prediction succeeded"),
    },
    {
      name: "Chain",
      tone: chain.up && chainDbSync.ok !== false ? "green" : chain.up ? "amber" : "red",
      state: !chain.up ? "down" : chainDbSync.ok === false ? "out of sync" : "ok",
      detail: chain.up
        ? `Block ${chain.blockNumber ?? "?"}, ${chain.tradeCount ?? 0} trades on-chain.${syncNote}`
        : (chain.error ?? "Unreachable"),
    },
  ];
}


export function HealthBadges({ health }: { health: Resource<HealthResponse> }) {
  const pills = buildPills(health);
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Connection status">
      {pills.map((p) => (
        <Badge key={p.name} tone={p.tone} title={p.detail}>
          <span aria-hidden>{p.tone === "green" ? "●" : p.tone === "red" ? "✕" : "○"}</span>
          {p.name}: {p.state}
        </Badge>
      ))}
      <button
        type="button"
        onClick={health.refetch}
        className="rounded-md border border-slate-400 bg-white px-2 py-0.5 text-sm font-medium text-slate-800 hover:bg-slate-100"
        title="Re-check connections"
      >
        Refresh
      </button>
    </div>
  );
}



