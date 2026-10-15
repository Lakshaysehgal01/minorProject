import { useMemo, useState } from "react";
import type { Resource } from "../hooks/useResource";
import type { Household } from "../types/api";
import { fmtSigned } from "../utils/format";
import { AsyncState } from "./AsyncState";
import { Badge } from "./Badge";
import { TIER_TONE } from "./HouseholdPicker";
import { Section } from "./Section";


type SortKey = "householdId" | "tier" | "panelCapacityKw" | "energyBalanceKwh" | "tokenBalance";


const COLUMNS: { key: SortKey; label: string; numeric: boolean }[] = [
  { key: "householdId", label: "Household", numeric: false },
  { key: "tier", label: "Tier", numeric: false },
  { key: "panelCapacityKw", label: "Panels (kW)", numeric: true },
  { key: "energyBalanceKwh", label: "Energy balance (kWh)", numeric: true },
  { key: "tokenBalance", label: "Token balance (credits)", numeric: true },
];


function signedCell(v: number) {
  const tone = v > 0 ? "text-emerald-800" : v < 0 ? "text-red-800" : "text-slate-700";
  return <span className={`font-semibold tabular-nums ${tone}`}>{fmtSigned(v)}</span>;
}


export function BalancesTable({ households, selected }: { households: Resource<Household[]>; selected: number }) {
  const [sortKey, setSortKey] = useState<SortKey>("householdId");
  const [asc, setAsc] = useState(true);


  const rows = useMemo(() => {
    if (!households.data) return [];
    const dir = asc ? 1 : -1;
    return [...households.data].sort((a, b) => {
      const x = a[sortKey];
      const y = b[sortKey];
      return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir || a.householdId - b.householdId;
    });
  }, [households.data, sortKey, asc]);


  function toggle(key: SortKey) {
    if (key === sortKey) setAsc((a) => !a);
    else {
      setSortKey(key);
      setAsc(true);
    }
  }


  return (
    <Section title="On-chain balances" subtitle="Read from the EnergyTrading smart contract. Click a column header to sort. Energy: + net buyer, − net seller.">
      <AsyncState resource={households} isEmpty={(d) => d.length === 0} emptyMessage="No households returned.">
        {() => (
          <div className="overflow-x-auto rounded-lg border border-slate-300">
            <table className="w-full text-left text-base">
              <thead className="bg-slate-100">
                <tr>
                  {COLUMNS.map((c) => (
                    <th key={c.key} scope="col" aria-sort={c.key === sortKey ? (asc ? "ascending" : "descending") : "none"} className={`px-3 py-2 ${c.numeric ? "text-right" : ""}`}>
                      <button type="button" onClick={() => toggle(c.key)} className="font-bold text-slate-900 hover:underline">
                        {c.label} {c.key === sortKey ? (asc ? "▲" : "▼") : ""}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((h) => (
                  <tr key={h.householdId} className={`border-t border-slate-200 ${h.householdId === selected ? "bg-blue-50" : ""}`}>
                    <td className="px-3 py-1.5 font-semibold">#{h.householdId}</td>
                    <td className="px-3 py-1.5"><Badge tone={TIER_TONE[h.tier]}>{h.tier}</Badge></td>
                    <td className="px-3 py-1.5 text-right tabular-nums">{h.panelCapacityKw.toFixed(2)}</td>
                    <td className="px-3 py-1.5 text-right">{signedCell(h.energyBalanceKwh)}</td>
                    <td className="px-3 py-1.5 text-right">{signedCell(h.tokenBalance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </AsyncState>
    </Section>
  );
}



