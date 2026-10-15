import type { ReactNode } from "react";
import type { Resource } from "../hooks/useResource";
import type { DailySummary } from "../types/api";
import { fmtKwh, fmtPct, fmtRs } from "../utils/format";
import { AsyncState } from "./AsyncState";


function KpiCard({ title, children, footnote }: { title: string; children: ReactNode; footnote?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col rounded-xl border border-slate-300 bg-white p-4 shadow-sm">
      <h3 className="text-sm font-bold tracking-wide text-slate-700 uppercase">{title}</h3>
      <div className="mt-2 flex-1 space-y-1">{children}</div>
      {footnote && <p className="mt-3 border-t border-slate-200 pt-2 text-sm text-slate-700">{footnote}</p>}
    </div>
  );
}


function Pair({ label, value, unit = "kWh", strong = false }: { label: string; value: string; unit?: string; strong?: boolean }) {
  return (
    <p className="flex items-baseline justify-between gap-2">
      <span className="text-base text-slate-700">{label}</span>
      <span className={`tabular-nums ${strong ? "text-2xl font-extrabold" : "text-xl font-bold"} text-slate-900`}>
        {value} <span className="text-sm font-medium text-slate-700">{unit}</span>
      </span>
    </p>
  );
}


export function KpiRow({ summary }: { summary: Resource<DailySummary> }) {
  return (
    <AsyncState resource={summary} notFoundMessage="Not simulated yet for this date. Click “Run Next Day” to simulate it.">
      {(s) => {
        const baseline = s.savings.gridOnlyBaselineCost;
        const pctVsBaseline = baseline > 0 ? (s.savings.realizedSavings / baseline) * 100 : null;
        return (
          <div>
            {!s.complete && (
              <p className="mb-3 rounded-lg border border-amber-600 bg-amber-50 px-3 py-2 text-base font-medium text-amber-900">
                Partial day: {s.slotsCovered} of 24 hours simulated so far.
              </p>
            )}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
              <KpiCard title="Solar generation">
                <Pair label="Predicted" value={fmtKwh(s.solar.predictedKwh)} />
                <Pair label="Actual" value={fmtKwh(s.solar.actualKwh)} />
              </KpiCard>
              <KpiCard title="Consumption">
                <Pair label="Predicted" value={fmtKwh(s.consumption.predictedKwh)} />
                <Pair label="Actual" value={fmtKwh(s.consumption.actualKwh)} />
              </KpiCard>
              <KpiCard
                title="Energy traded P2P"
                footnote={s.trading.failedTradeCount > 0 ? `${s.trading.failedTradeCount} failed trade(s) excluded` : "All trades settled on-chain"}
              >
                <Pair label="Energy" value={fmtKwh(s.trading.tradedKwh)} strong />
                <Pair label="Trades" value={String(s.trading.tradeCount)} unit="trades" />
              </KpiCard>
              <KpiCard
                title="Deficit demand met by P2P"
                footnote="No storage: trades only happen when some homes have a surplus while others have a deficit."
              >
                <Pair label="Of predicted deficit" value={fmtPct(s.trading.percentDeficitMetP2P)} unit="" strong />
                <Pair label="Of actual deficit" value={fmtPct(s.trading.percentActualDeficitMetP2P)} unit="" />
              </KpiCard>
              <KpiCard
                title="Simulated savings (assumed tariffs)"
                footnote={`Gross ${fmtRs(s.savings.grossSavings)} · perfect-foresight bound ${fmtRs(s.savings.perfectForesightSavings)}`}
              >
                <Pair label="Realized" value={fmtRs(s.savings.realizedSavings)} unit="" strong />
                <Pair label="Grid-only baseline" value={fmtRs(baseline)} unit="" />
                <p className="text-base font-semibold text-slate-900">
                  {pctVsBaseline === null ? "Savings % vs baseline: n/a" : `${fmtPct(pctVsBaseline)} below grid-only baseline`}
                </p>
              </KpiCard>
            </div>
          </div>
        );
      }}
    </AsyncState>
  );
}



