import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HourPoint } from "../utils/chartData";
import { HourTooltip } from "./HourTooltip";


export const COLORS = {
  predicted: "#1d4ed8",
  actual: "#c2410c",
  surplus: "#15803d",
  deficit: "#b91c1c",
  anomaly: "#f59e0b",
};


type Kind = "solar" | "consumption";


const CONFIG = {
  solar: { predicted: "predictedSolar", actual: "actualSolar", anomaly: "solarAnomaly", label: "Solar generation (kWh)" },
  consumption: {
    predicted: "predictedConsumption",
    actual: "actualConsumption",
    anomaly: "consumptionAnomaly",
    label: "Consumption (kWh)",
  },
} as const;


export function PredActualChart({ data, kind }: { data: HourPoint[]; kind: Kind }) {
  const cfg = CONFIG[kind];
  const anomalies = data.filter((d) => d[cfg.anomaly]);


  return (
    <div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
          <XAxis dataKey="label" interval={2} tick={{ fill: "#0f172a" }} label={{ value: "Hour of day (simulated local time)", position: "insideBottom", offset: -18, fill: "#0f172a" }} />
          <YAxis tick={{ fill: "#0f172a" }} label={{ value: "kWh", angle: -90, position: "insideLeft", fill: "#0f172a" }} />
          <Tooltip content={<HourTooltip kind={kind} />} />
          <Legend verticalAlign="top" height={32} />
          {anomalies.map((d) => (
            <ReferenceLine key={d.label} x={d.label} stroke={COLORS.anomaly} strokeWidth={24} strokeOpacity={0.35} ifOverflow="visible" />
          ))}
          <Line type="monotone" dataKey={cfg.predicted} name="Predicted" stroke={COLORS.predicted} strokeWidth={3} strokeDasharray="8 5" dot={false} connectNulls={false} />
          <Line type="monotone" dataKey={cfg.actual} name="Actual" stroke={COLORS.actual} strokeWidth={3} dot={{ r: 3 }} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
      {anomalies.length > 0 && (
        <p className="mt-1 text-base font-medium text-amber-900">
          <span className="mr-2 inline-block h-3 w-6 rounded-sm align-middle" style={{ background: COLORS.anomaly, opacity: 0.5 }} />
          Highlighted hour(s) ({anomalies.map((a) => a.label).join(", ")}): possible outage/fault in the data, not a model error.
        </p>
      )}
    </div>
  );
}



