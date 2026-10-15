import { useMemo } from "react";
import { Bar, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useForecast } from "../hooks/resources";
import { addDays, REPLAY_MAX } from "../utils/date";
import { forecastTotals, modelDisplayName, toForecastSeries } from "../utils/chartData";
import { fmtKwh, fmtSigned } from "../utils/format";
import { AsyncState } from "./AsyncState";
import { COLORS } from "./PredActualChart";
import { Section } from "./Section";


function Totals({ solar, consumption, net }: { solar: number; consumption: number; net: number }) {
  const items: [string, string, string][] = [
    ["Predicted generation", `${fmtKwh(solar)} kWh`, "text-slate-900"],
    ["Predicted consumption", `${fmtKwh(consumption)} kWh`, "text-slate-900"],
    ["Predicted net", `${fmtSigned(net)} kWh`, net >= 0 ? "text-emerald-800" : "text-red-800"],
  ];
  return (
    <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
      {items.map(([label, value, tone]) => (
        <div key={label} className="rounded-lg border border-slate-300 bg-slate-50 p-3">
          <p className="text-sm font-semibold text-slate-700">{label}</p>
          <p className={`text-2xl font-extrabold tabular-nums ${tone}`}>{value}</p>
        </div>
      ))}
    </div>
  );
}


export function ForecastPanel({ householdId, date }: { householdId: number; date: string | null }) {
  const forecastDate = date ? addDays(date, 1) : null;
  const beyondDataset = forecastDate !== null && forecastDate > REPLAY_MAX;
  const forecast = useForecast(householdId, beyondDataset ? null : forecastDate);
  const series = useMemo(() => (forecast.data ? toForecastSeries(forecast.data.hours) : []), [forecast.data]);
  const totals = useMemo(() => forecastTotals(series), [series]);


  return (
    <Section
      title={`Next Day Forecast${forecastDate ? `: ${forecastDate}` : ""}`}
      subtitle={`Household #${householdId}. Uses the dataset's weather columns as input; a real deployment would use a weather forecast API.`}
    >
      {beyondDataset ? (
        <p className="rounded-lg border border-slate-300 bg-slate-50 p-4 text-base text-slate-800">
          No forecast available: the dataset ends on {REPLAY_MAX}.
        </p>
      ) : (
        <AsyncState resource={forecast} notFoundMessage="The dataset has no weather data for the forecast date.">
          {() => (
            <>
              {forecast.data && (
                <p className="mb-3 text-base font-semibold text-slate-900">
                  Predicted by: solar → {modelDisplayName(forecast.data.models.solar.model)}, consumption →{" "}
                  {modelDisplayName(forecast.data.models.consumption.model)}
                  {forecast.data.inSample && " (training-period date: in-sample)"}
                </p>
              )}
              <Totals {...totals} />
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={series} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                  <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                  <XAxis dataKey="label" interval={2} tick={{ fill: "#0f172a" }} label={{ value: "Hour of day (simulated local time)", position: "insideBottom", offset: -18, fill: "#0f172a" }} />
                  <YAxis tick={{ fill: "#0f172a" }} label={{ value: "kWh", angle: -90, position: "insideLeft", fill: "#0f172a" }} />
                  <Tooltip formatter={(v) => (typeof v === "number" ? `${fmtKwh(v)} kWh` : String(v))} />
                  <Legend verticalAlign="top" height={32} />
                  <ReferenceLine y={0} stroke="#0f172a" />
                  <Bar dataKey="net" name="Predicted net" isAnimationActive={false}>
                    {series.map((p) => (
                      <Cell key={p.label} fill={p.net >= 0 ? COLORS.surplus : COLORS.deficit} fillOpacity={0.55} />
                    ))}
                  </Bar>
                  <Line type="monotone" dataKey="solar" name="Predicted generation" stroke={COLORS.predicted} strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="consumption" name="Predicted consumption" stroke={COLORS.actual} strokeWidth={3} strokeDasharray="8 5" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
              <details className="mt-3">
                <summary className="cursor-pointer text-base font-semibold text-blue-800">Show hourly values</summary>
                <div className="mt-2 max-h-72 overflow-auto rounded-lg border border-slate-300">
                  <table className="w-full text-left text-base">
                    <thead className="sticky top-0 bg-slate-100">
                      <tr>
                        <th className="px-3 py-1.5">Hour</th>
                        <th className="px-3 py-1.5 text-right">Generation (kWh)</th>
                        <th className="px-3 py-1.5 text-right">Consumption (kWh)</th>
                        <th className="px-3 py-1.5 text-right">Net (kWh)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {series.map((p) => (
                        <tr key={p.label} className="border-t border-slate-200">
                          <td className="px-3 py-1">{p.label}</td>
                          <td className="px-3 py-1 text-right tabular-nums">{fmtKwh(p.solar)}</td>
                          <td className="px-3 py-1 text-right tabular-nums">{fmtKwh(p.consumption)}</td>
                          <td className={`px-3 py-1 text-right font-semibold tabular-nums ${p.net >= 0 ? "text-emerald-800" : "text-red-800"}`}>{fmtSigned(p.net)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          )}
        </AsyncState>
      )}
    </Section>
  );
}



