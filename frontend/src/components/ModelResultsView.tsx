import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelTable as ModelTableData } from "../types/api";
import { isActiveRow, isBenchmarkRow, modelLabel, r2Domain, sortByR2 } from "../utils/chartData";
import { Badge } from "./Badge";
import { Section } from "./Section";


const BEST = "#15803d";
const OTHER = "#475569";
const BENCH = "#94a3b8";


export function ModelTable({ rows, active }: { rows: ModelTableData["rows"]; active?: string }) {
  const sorted = useMemo(() => sortByR2(rows), [rows]);
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-300">
      <table className="w-full text-left text-base">
        <thead className="bg-slate-100 text-sm font-bold text-slate-800 uppercase">
          <tr>
            <th className="px-3 py-2">Model</th>
            <th className="px-3 py-2 text-right">RMSE</th>
            <th className="px-3 py-2 text-right">MAE</th>
            <th className="px-3 py-2 text-right">R²</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.Model} className={`border-t border-slate-200 ${i === 0 ? "bg-emerald-100 font-bold" : ""}`}>
              <td className="px-3 py-1.5">
                {modelLabel(r.Model)} {i === 0 && <Badge tone="green">★ best R²</Badge>}{" "}
                {isActiveRow(r.Model, active) && <Badge tone="blue">▶ in use</Badge>}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.RMSE.toFixed(3)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.MAE.toFixed(3)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.R2.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


export function R2Chart({ rows, label }: { rows: ModelTableData["rows"]; label: string }) {
  const data = useMemo(() => sortByR2(rows).map((r, i) => ({ name: modelLabel(r.Model), R2: r.R2, best: i === 0, bench: isBenchmarkRow(r.Model) })), [rows]);
  const domain = useMemo(() => r2Domain(rows), [rows]);
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 40 + 60)}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 60, bottom: 40, left: 10 }}>
        <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={domain}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fill: "#0f172a" }}
          label={{ value: `R² on test period: ${label} (axis starts at ${domain[0].toFixed(2)}, not 0)`, position: "insideBottom", offset: -20, fill: "#0f172a" }}
        />
        <YAxis type="category" dataKey="name" width={230} tick={{ fill: "#0f172a" }} />
        <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))} />
        <Bar dataKey="R2" name="R²" isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.name} fill={d.best ? BEST : d.bench ? BENCH : OTHER} />
          ))}
          <LabelList dataKey="R2" position="right" formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} fill="#0f172a" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}


export function TargetSection({ title, table, active }: { title: string; table: ModelTableData; active?: string }) {
  const hasBenchmarks = table.rows.some((r) => isBenchmarkRow(r.Model));
  return (
    <Section
      title={title}
      subtitle={`From ${table.file}, generated ${table.generatedAt.slice(0, 16).replace("T", " ")} UTC`}
    >
      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-2">
        <div className="min-w-0">
          <ModelTable rows={table.rows} active={active} />
          {hasBenchmarks && (
            <p className="mt-2 text-sm text-slate-700">* Prophet rows are single-household benchmarks, not the pooled models the API serves.</p>
          )}
        </div>
        <div className="min-w-0">
          <R2Chart rows={table.rows} label={title} />
        </div>
      </div>
    </Section>
  );
}



