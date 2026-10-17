import { useMemo } from "react";
import { AsyncState } from "../components/AsyncState";
import { BalancesTable } from "../components/BalancesTable";
import { ForecastPanel } from "../components/ForecastPanel";
import { HealthBadges } from "../components/HealthBadges";
import { HouseholdPicker } from "../components/HouseholdPicker";
import { KpiRow } from "../components/KpiRow";
import { NetChart } from "../components/NetChart";
import { PredActualChart } from "../components/PredActualChart";
import { Section } from "../components/Section";
import { TradesTable } from "../components/TradesTable";
import { TradesNetworkGraph } from "../components/TradesNetworkGraph";
import { usePredictions, useSummary } from "../hooks/resources";
import type { Resource } from "../hooks/useResource";
import type { HealthResponse, Household } from "../types/api";
import { toHourSeries } from "../utils/chartData";


interface Props {
  date: string | null;
  householdId: number;
  onHouseholdChange: (id: number) => void;
  households: Resource<Household[]>;
  health: Resource<HealthResponse>;
}


const NOT_SIMULATED = "Not simulated yet for this date. Click \u201cRun Next Day\u201d to simulate it.";


export function DashboardPage({ date, householdId, onHouseholdChange, households, health }: Props) {
  const summary = useSummary(date);
  const predictions = usePredictions(householdId, date);
  const series = useMemo(() => (predictions.data ? toHourSeries(predictions.data.hours) : []), [predictions.data]);


  const chartSubtitle = `Household #${householdId}${date ? `, ${date}` : ""}`;


  return (
    <div className="space-y-6">
      {/* Service status strip — shows ML model in use, chain, backend */}
      <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Service status</p>
        <HealthBadges health={health} />
      </div>

      <Section title={`Daily summary${date ? `: ${date}` : ""}`} subtitle="Simulated from replayed meter data; predictions come from the live ML service.">
        <KpiRow summary={summary} />
      </Section>


      <HouseholdPicker households={households} selected={householdId} onSelect={onHouseholdChange} />


      <Section title="Solar generation: predicted vs actual" subtitle={chartSubtitle}>
        <AsyncState resource={predictions} notFoundMessage={NOT_SIMULATED}>
          {() => <PredActualChart data={series} kind="solar" />}
        </AsyncState>
      </Section>


      <Section title="Consumption: predicted vs actual" subtitle={chartSubtitle}>
        <AsyncState resource={predictions} notFoundMessage={NOT_SIMULATED}>
          {() => <PredActualChart data={series} kind="consumption" />}
        </AsyncState>
      </Section>


      <Section title="Net surplus / deficit per hour" subtitle={`${chartSubtitle}. Predicted net = predicted solar minus predicted consumption (what the market traded on).`}>
        <AsyncState resource={predictions} notFoundMessage={NOT_SIMULATED}>
          {() => <NetChart data={series} />}
        </AsyncState>
      </Section>


      <ForecastPanel householdId={householdId} date={date} />


      <TradesTable date={date} householdId={householdId} />

      <TradesNetworkGraph
        date={date}
        householdId={householdId}
        onSelectHousehold={onHouseholdChange}
        households={households}
      />

      <BalancesTable households={households} selected={householdId} />
    </div>
  );
}



