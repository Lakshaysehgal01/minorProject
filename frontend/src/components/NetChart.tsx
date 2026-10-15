import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HourPoint } from "../utils/chartData";
import { COLORS } from "./PredActualChart";
import { HourTooltip } from "./HourTooltip";


function NetLegend() {
  const items: [string, string][] = [
    [COLORS.surplus, "Surplus (can sell)"],
    [COLORS.deficit, "Deficit (needs to buy)"],
  ];
  return (
    <ul className="m-0 flex list-none justify-center gap-6 p-0 text-base text-slate-900">
      {items.map(([color, label]) => (
        <li key={label} className="flex items-center gap-2">
          <span className="inline-block h-4 w-4 rounded-sm" style={{ background: color }} />
          {label}
        </li>
      ))}
    </ul>
  );
}


export function NetChart({ data }: { data: HourPoint[] }) {
  const anomalies = data.filter((d) => d.solarAnomaly || d.consumptionAnomaly);
  return (
    <div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
          <XAxis dataKey="label" interval={2} tick={{ fill: "#0f172a" }} label={{ value: "Hour of day (simulated local time)", position: "insideBottom", offset: -18, fill: "#0f172a" }} />
          <YAxis tick={{ fill: "#0f172a" }} label={{ value: "Net kWh (+ surplus / − deficit)", angle: -90, position: "insideLeft", fill: "#0f172a", style: { textAnchor: "middle" } }} />
          <Tooltip content={<HourTooltip kind="net" />} />
          <Legend verticalAlign="top" height={32} content={NetLegend} />
          {anomalies.map((d) => (
            <ReferenceLine key={d.label} x={d.label} stroke={COLORS.anomaly} strokeWidth={24} strokeOpacity={0.35} ifOverflow="visible" />
          ))}
          <ReferenceLine y={0} stroke="#0f172a" strokeWidth={2} />
          <Bar dataKey="predictedNet" name="Predicted net" isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.label} fill={(d.predictedNet ?? 0) >= 0 ? COLORS.surplus : COLORS.deficit} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {anomalies.length > 0 && (
        <p className="mt-1 text-base font-medium text-amber-900">
          Highlighted hour(s) ({anomalies.map((a) => a.label).join(", ")}): possible outage/fault in the data.
        </p>
      )}
    </div>
  );
}



