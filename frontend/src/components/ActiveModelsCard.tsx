import type { Resource } from "../hooks/useResource";
import type { ActiveModel, ActiveModels } from "../types/api";
import { modelDisplayName } from "../utils/chartData";
import { AsyncState } from "./AsyncState";
import { Badge } from "./Badge";
import { Section } from "./Section";


export function ModelChip({ target, m }: { target: string; m: ActiveModel }) {
  return (
    <div className="rounded-lg border border-slate-300 bg-slate-50 p-3">
      <p className="text-sm font-bold text-slate-700 uppercase">{target}</p>
      <p className="mt-1 text-2xl font-extrabold text-slate-900">{modelDisplayName(m.model)}</p>
      <p className="mt-1 flex flex-wrap items-center gap-2 text-base text-slate-800">
        <Badge tone={m.source === "auto" ? "green" : "amber"}>{m.source === "auto" ? "auto-selected" : "pinned in .env"}</Badge>
        {m.r2 !== null && <span>test R² {m.r2.toFixed(4)}</span>}
      </p>
    </div>
  );
}


// Which model the backend calls for every prediction, including the next-day forecast.
export function ActiveModelsCard({ models }: { models: Resource<ActiveModels> }) {
  return (
    <Section
      title="Models in use"
      subtitle="Called for every prediction the backend makes: the hourly simulation and the Next Day Forecast. With ML_MODEL=auto the highest-R² model in the tables below is chosen separately for each target when the backend starts."
    >
      <AsyncState resource={models}>
        {(m) => (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ModelChip target="Solar generation" m={m.solar} />
            <ModelChip target="Consumption" m={m.consumption} />
          </div>
        )}
      </AsyncState>
    </Section>
  );
}



