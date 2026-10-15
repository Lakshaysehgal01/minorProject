import { ActiveModelsCard } from "../components/ActiveModelsCard";
import { OverallStatsCard } from "../components/OverallStatsCard";
import { AsyncState } from "../components/AsyncState";
import { TargetSection } from "../components/ModelResultsView";
import { useActiveModels, useModelResults } from "../hooks/resources";
import type { Resource } from "../hooks/useResource";
import type { Stats } from "../types/api";


const NO_RESULTS = "Run the ML training and model_comparison.py to generate results.";


export function ModelResultsPage({ stats }: { stats: Resource<Stats> }) {
  const results = useModelResults();
  const active = useActiveModels();
  return (
    <div className="space-y-6">
      <ActiveModelsCard models={active} />
      <OverallStatsCard stats={stats} />
      <AsyncState
        resource={results}
        notFoundMessage={NO_RESULTS}
        isEmpty={(r) => r.solar.rows.length === 0 && r.consumption.rows.length === 0}
        emptyMessage={NO_RESULTS}
      >
        {(r) => (
          <>
            <TargetSection title="Solar generation models" table={r.solar} active={active.data?.solar.model} />
            <TargetSection title="Consumption models" table={r.consumption} active={active.data?.consumption.model} />
          </>
        )}
      </AsyncState>
    </div>
  );
}



