import { useMemo } from "react";
import { AsyncState } from "../components/AsyncState";
import { BalancesTable } from "../components/BalancesTable";
import { ForecastPanel } from "../components/ForecastPanel";
import { HouseholdPicker } from "../components/HouseholdPicker";
import { KpiRow } from "../components/KpiRow";
import { NetChart } from "../components/NetChart";
import { PredActualChart } from "../components/PredActualChart";
import { Section } from "../components/Section";
import { TradesTable } from "../components/TradesTable";
import { usePredictions, useSummary } from "../hooks/resources";
import type { Resource } from "../hooks/useResource";
import type { Household } from "../types/api";
import { toHourSeries } from "../utils/chartData";


interface Props {
  date: string | null;
  householdId: number;
  onHouseholdChange: (id: number) => void;
  households: Resource<Household[]>;
}


const NOT_SIMULATED = "Not simulated yet for this date. Click “Run Next Day” to simulate it.";


export function DashboardPage({ date, householdId, onHouseholdChange, households }: Props) {
  const summary = useSummary(date);
  const predictions = usePredictions(householdId, date);
  const series = useMemo(() => (predictions.data ? toHourSeries(predictions.data.hours) : []), [predictions.data]);


  const chartSubtitle = `Household #${householdId}${date ? `, ${date}` : ""}`;


  return (
    <div className="space-y-6">
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


      <BalancesTable households={households} selected={householdId} />
    </div>
  );
}



