import { useEffect, useMemo, useState } from "react";
import { useHouseholdTrades, useTrades } from "../hooks/resources";
import type { Resource } from "../hooks/useResource";
import type { Trade } from "../types/api";
import { fmtNum, fmtRs, shortHash } from "../utils/format";
import { AsyncState } from "./AsyncState";
import { Badge } from "./Badge";
import { CopyButton } from "./CopyButton";
import { Section } from "./Section";


const PAGE_SIZE = 10;


interface TradeList {
  trades: Trade[];
  total: number;
}


function StatusBadge({ trade }: { trade: Trade }) {
  if (trade.status === "confirmed") return <Badge tone="green">✓ confirmed</Badge>;
  if (trade.status === "failed") return <Badge tone="red" title={trade.error}>✕ failed</Badge>;
  return <Badge tone="amber">… pending</Badge>;
}


function Table({ list, page, onPage }: { list: TradeList; page: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(list.total / PAGE_SIZE));
  return (
    <>
      <div className="overflow-x-auto rounded-lg border border-slate-300">
        <table className="w-full text-left text-base">
          <thead className="bg-slate-100 text-sm font-bold text-slate-800 uppercase">
            <tr>
              <th className="px-3 py-2">Time</th>
              <th className="px-3 py-2">Seller</th>
              <th className="px-3 py-2">Buyer</th>
              <th className="px-3 py-2 text-right">kWh</th>
              <th className="px-3 py-2 text-right">Price (Rs./kWh)</th>
              <th className="px-3 py-2 text-right">Value</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Tx hash</th>
            </tr>
          </thead>
          <tbody>
            {list.trades.map((t) => (
              <tr key={t._id} className="border-t border-slate-200">
                <td className="px-3 py-1.5 tabular-nums">{t.slotTimestamp.slice(11, 16)}</td>
                <td className="px-3 py-1.5">#{t.sellerId}</td>
                <td className="px-3 py-1.5">#{t.buyerId}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmtNum(t.kWh)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmtNum(t.pricePerKwh)}</td>
                <td className="px-3 py-1.5 text-right font-semibold tabular-nums">{fmtRs(t.costMilli / 1000)}</td>
                <td className="px-3 py-1.5"><StatusBadge trade={t} /></td>
                <td className="px-3 py-1.5">
                  {t.txHash ? (
                    <span className="inline-flex items-center gap-2">
                      <code title={t.txHash} className="font-mono text-base">{shortHash(t.txHash)}</code>
                      <CopyButton text={t.txHash} />
                    </span>
                  ) : (
                    <span className="text-slate-600">n/a</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-base">
        <span className="text-slate-800">
          {list.total} trade(s) · page {page} of {pages} · newest first
        </span>
        <span className="flex gap-2">
          <button type="button" onClick={() => onPage(page - 1)} disabled={page <= 1} className="rounded-md border-2 border-slate-500 bg-white px-3 py-1 font-semibold disabled:border-slate-300 disabled:text-slate-400">
            Previous
          </button>
          <button type="button" onClick={() => onPage(page + 1)} disabled={page >= pages} className="rounded-md border-2 border-slate-500 bg-white px-3 py-1 font-semibold disabled:border-slate-300 disabled:text-slate-400">
            Next
          </button>
        </span>
      </div>
    </>
  );
}


export function TradesTable({ date, householdId }: { date: string | null; householdId: number }) {
  const [onlyHousehold, setOnlyHousehold] = useState(false);
  const [page, setPage] = useState(1);
  useEffect(() => setPage(1), [date, householdId, onlyHousehold]);


  const serverTrades = useTrades(date, page, PAGE_SIZE, !onlyHousehold);
  const householdTrades = useHouseholdTrades(householdId, onlyHousehold);


  // /trades/household/:id has no date filter, so filter by the selected date and paginate here.
  const householdList = useMemo<Resource<TradeList>>(() => {
    const forDate = householdTrades.data?.filter((t) => t.slotTimestamp.slice(0, 10) === date) ?? null;
    return {
      ...householdTrades,
      data: forDate && { total: forDate.length, trades: forDate.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE) },
    };
  }, [householdTrades, date, page]);


  const serverList = useMemo<Resource<TradeList>>(
    () => ({ ...serverTrades, data: serverTrades.data && { total: serverTrades.data.total, trades: serverTrades.data.trades } }),
    [serverTrades]
  );


  return (
    <Section
      title={`Trades${date ? ` on ${date}` : ""}`}
      subtitle="Settled on a local Hardhat chain (no block explorer, so hashes are shown for reference and copying only)."
      actions={
        <label className="flex cursor-pointer items-center gap-2 text-base font-semibold text-slate-900">
          <input type="checkbox" checked={onlyHousehold} onChange={(e) => setOnlyHousehold(e.target.checked)} className="h-5 w-5" />
          Only household #{householdId}
        </label>
      }
    >
      <AsyncState
        resource={onlyHousehold ? householdList : serverList}
        notFoundMessage="Not simulated yet for this date. Click “Run Next Day” to simulate it."
        isEmpty={(d) => d.total === 0}
        emptyMessage={onlyHousehold ? `Household #${householdId} made no trades on this date.` : "No trades on this date (trades only occur when some homes have a surplus while others have a deficit)."}
      >
        {(list) => <Table list={list} page={page} onPage={setPage} />}
      </AsyncState>
    </Section>
  );
}



