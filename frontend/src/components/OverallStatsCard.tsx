import type { Resource } from "../hooks/useResource";
import type { MaeResult, Stats } from "../types/api";
import { fmtInt, fmtKwh, fmtNum, fmtRs } from "../utils/format";
import { AsyncState } from "./AsyncState";
import { Section } from "./Section";


function MaeBlock({ title, mae }: { title: string; mae: MaeResult }) {
  return (
    <div className="rounded-lg border border-slate-300 bg-slate-50 p-3">
      <p className="text-sm font-bold text-slate-700 uppercase">{title}</p>
      <p className="mt-1 text-base text-slate-800">
        All hours: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtNum(mae.mae, 3)}</span> kWh
      </p>
      <p className="text-base text-slate-800">
        Excluding anomaly hours: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtNum(mae.maeExcludingAnomalies, 3)}</span> kWh
      </p>
      <p className="text-sm text-slate-700">{fmtInt(mae.nAnomalies)} anomaly hour(s) of {fmtInt(mae.n)} household-hours</p>
    </div>
  );
}


export function OverallStatsCard({ stats }: { stats: Resource<Stats> }) {
  return (
    <Section title="Overall simulation stats" subtitle="Everything simulated so far, from GET /stats. Anomaly hours are outage/fault hours in the dataset, not model errors.">
      <AsyncState resource={stats} isEmpty={(s) => s.simulation.slotsCompleted === 0} emptyMessage="No simulation yet. Click “Run Next Day” on the Dashboard tab.">
        {(s) => (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-4">
            <MaeBlock title="Solar prediction MAE" mae={s.predictionError.solar} />
            <MaeBlock title="Consumption prediction MAE" mae={s.predictionError.consumption} />
            <div className="rounded-lg border border-slate-300 bg-slate-50 p-3">
              <p className="text-sm font-bold text-slate-700 uppercase">Trading</p>
              <p className="mt-1 text-base text-slate-800">
                Total trades: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtInt(s.trading.tradeCount)}</span>
              </p>
              <p className="text-base text-slate-800">
                Energy traded: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtKwh(s.trading.tradedKwh)}</span> kWh
              </p>
              <p className="text-sm text-slate-700">{fmtInt(s.simulation.slotsCompleted)} of {fmtInt(s.simulation.totalSlots)} hours simulated</p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-slate-50 p-3">
              <p className="text-sm font-bold text-slate-700 uppercase">Simulated savings (assumed tariffs)</p>
              <p className="mt-1 text-base text-slate-800">
                Realized: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtRs(s.savings.realizedSavings)}</span>
              </p>
              <p className="text-base text-slate-800">
                Gross: <span className="text-xl font-extrabold tabular-nums text-slate-900">{fmtRs(s.savings.grossSavings)}</span>
              </p>
              <p className="text-sm text-slate-700">Perfect-foresight bound {fmtRs(s.savings.perfectForesightSavings)}</p>
            </div>
          </div>
        )}
      </AsyncState>
    </Section>
  );
}



