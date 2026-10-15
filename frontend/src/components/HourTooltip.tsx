import type { HourPoint } from "../utils/chartData";
import { fmtKwh, fmtSigned } from "../utils/format";


export type TooltipKind = "solar" | "consumption" | "net";


interface Props {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: HourPoint }>;
  kind: TooltipKind;
}


// recharts injects `active` and `payload`; `kind` is ours.
export function HourTooltip({ active, payload, kind }: Props) {
  const p = payload?.[0]?.payload;
  if (!active || !p) return null;


  const anomaly = kind === "solar" ? p.solarAnomaly : kind === "consumption" ? p.consumptionAnomaly : p.solarAnomaly || p.consumptionAnomaly;
  const rows: [string, string][] =
    kind === "solar"
      ? [["Predicted", `${fmtKwh(p.predictedSolar)} kWh`], ["Actual", `${fmtKwh(p.actualSolar)} kWh`]]
      : kind === "consumption"
        ? [["Predicted", `${fmtKwh(p.predictedConsumption)} kWh`], ["Actual", `${fmtKwh(p.actualConsumption)} kWh`]]
        : [
            ["Predicted net", p.predictedNet === undefined ? "n/a" : `${fmtSigned(p.predictedNet)} kWh`],
            ["Actual net", p.actualNet === undefined ? "n/a" : `${fmtSigned(p.actualNet)} kWh`],
          ];


  return (
    <div className="rounded-lg border-2 border-slate-400 bg-white p-3 text-base shadow-lg">
      <p className="mb-1 font-bold">{p.label}</p>
      {rows.map(([k, v]) => (
        <p key={k} className="flex justify-between gap-4">
          <span className="text-slate-700">{k}</span>
          <span className="font-semibold tabular-nums">{v}</span>
        </p>
      ))}
      {anomaly && <p className="mt-2 max-w-56 font-semibold text-amber-800">Possible outage/fault in data (not a model error)</p>}
    </div>
  );
}



