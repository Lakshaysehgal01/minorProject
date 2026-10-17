import { useMemo, useState, useEffect, useRef } from "react";
import { useAllTradesForDate } from "../hooks/resources";
import type { Resource } from "../hooks/useResource";
import type { Household, Trade } from "../types/api";
import { fmtKwh, fmtNum, fmtRs } from "../utils/format";
import { Section } from "./Section";

interface Props {
  date: string | null;
  householdId: number;
  onSelectHousehold: (id: number) => void;
  households: Resource<Household[]>;
}

interface AggregatedTradeEdge {
  id: string;
  sellerId: number;
  buyerId: number;
  totalKwh: number;
  totalCost: number;
  tradeCount: number;
  trades: Trade[];
}

export function TradesNetworkGraph({ date, householdId, onSelectHousehold, households }: Props) {
  const tradesResource = useAllTradesForDate(date);
  const [selectedHour, setSelectedHour] = useState<number | "all">("all");
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<AggregatedTradeEdge | null>(null);
  const [viewMode, setViewMode] = useState<"ring" | "hub">("ring");
  const [focusOnlySelected, setFocusOnlySelected] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Reset hour filter when date changes
  useEffect(() => {
    setSelectedHour("all");
    setIsPlaying(false);
  }, [date]);

  // Auto-play hours
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setSelectedHour((prev) => {
        if (prev === "all") return 0;
        const next = (prev + 1) % 24;
        return next;
      });
    }, 1200);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying]);

  const allTrades = tradesResource.data ?? [];

  // Filter trades by hour if specified
  const filteredTrades = useMemo(() => {
    if (selectedHour === "all") return allTrades;
    return allTrades.filter((t) => {
      const hourStr = t.slotTimestamp.slice(11, 13);
      return parseInt(hourStr, 10) === selectedHour;
    });
  }, [allTrades, selectedHour]);

  // Aggregate trades by (sellerId, buyerId)
  const edges = useMemo(() => {
    const map = new Map<string, AggregatedTradeEdge>();
    for (const t of filteredTrades) {
      if (t.status === "failed") continue;
      const key = `${t.sellerId}->${t.buyerId}`;
      const existing = map.get(key);
      const cost = t.costMilli / 1000;
      if (existing) {
        existing.totalKwh += t.kWh;
        existing.totalCost += cost;
        existing.tradeCount += 1;
        existing.trades.push(t);
      } else {
        map.set(key, {
          id: key,
          sellerId: t.sellerId,
          buyerId: t.buyerId,
          totalKwh: t.kWh,
          totalCost: cost,
          tradeCount: 1,
          trades: [t],
        });
      }
    }
    return Array.from(map.values());
  }, [filteredTrades]);

  // Microgrid stats for the current filter
  const stats = useMemo(() => {
    let totalKwh = 0;
    let totalCost = 0;
    let selectedIncomingKwh = 0;
    let selectedOutgoingKwh = 0;
    let selectedIncomingCost = 0;
    let selectedOutgoingCost = 0;
    let selectedIncomingCount = 0;
    let selectedOutgoingCount = 0;

    for (const e of edges) {
      totalKwh += e.totalKwh;
      totalCost += e.totalCost;
      if (e.buyerId === householdId) {
        selectedIncomingKwh += e.totalKwh;
        selectedIncomingCost += e.totalCost;
        selectedIncomingCount += e.tradeCount;
      }
      if (e.sellerId === householdId) {
        selectedOutgoingKwh += e.totalKwh;
        selectedOutgoingCost += e.totalCost;
        selectedOutgoingCount += e.tradeCount;
      }
    }

    return {
      totalKwh,
      totalCost,
      totalEdges: edges.length,
      selectedIncomingKwh,
      selectedOutgoingKwh,
      selectedIncomingCost,
      selectedOutgoingCost,
      selectedIncomingCount,
      selectedOutgoingCount,
      netKwh: selectedOutgoingKwh - selectedIncomingKwh,
      netCost: selectedOutgoingCost - selectedIncomingCost,
    };
  }, [edges, householdId]);

  // Households list (default 25 households 0..24)
  const householdList = useMemo(() => {
    if (households.data && households.data.length > 0) return households.data;
    return Array.from({ length: 25 }, (_, i) => ({
      householdId: i,
      tier: (i % 3 === 0 ? "large" : i % 2 === 0 ? "medium" : "small") as "small" | "medium" | "large",
      panelCapacityKw: 4,
      energyBalanceMilli: 0,
      tokenBalanceMilli: 0,
      energyBalanceKwh: 0,
      tokenBalance: 0,
    }));
  }, [households.data]);

  // SVG dimensions
  const width = 760;
  const height = 560;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 220;

  // Compute node positions for Ring View
  const nodePositions = useMemo(() => {
    const total = householdList.length;
    const positions = new Map<number, { x: number; y: number }>();
    householdList.forEach((h, index) => {
      const angle = (2 * Math.PI * index) / total - Math.PI / 2;
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);
      positions.set(h.householdId, { x, y });
    });
    return positions;
  }, [householdList, cx, cy, radius]);

  // Compute node positions for Hub & Spoke View
  const hubPositions = useMemo(() => {
    const positions = new Map<number, { x: number; y: number }>();
    positions.set(householdId, { x: cx, y: cy });

    // Find peers trading with selected household
    const partners = new Set<number>();
    edges.forEach((e) => {
      if (e.sellerId === householdId) partners.add(e.buyerId);
      if (e.buyerId === householdId) partners.add(e.sellerId);
    });
    const partnerList = Array.from(partners);
    const partnerRadius = 180;
    const total = Math.max(partnerList.length, 1);

    partnerList.forEach((pid, idx) => {
      const angle = (2 * Math.PI * idx) / total - Math.PI / 2;
      positions.set(pid, {
        x: cx + partnerRadius * Math.cos(angle),
        y: cy + partnerRadius * Math.sin(angle),
      });
    });

    return positions;
  }, [householdId, edges, cx, cy]);

  const activeFocus = hoveredNode !== null ? hoveredNode : householdId;

  // Max edge kWh for stroke width scaling
  const maxKwh = useMemo(() => {
    let m = 0.1;
    for (const e of edges) if (e.totalKwh > m) m = e.totalKwh;
    return m;
  }, [edges]);

  // Hours options
  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <Section
      title="Peer-to-Peer Energy Trading Network Graph"
      subtitle={
        <span>
          Visual map of peer-to-peer energy transactions on {date ? date : "the selected date"}. Houses represent
          microgrid nodes. <span className="font-semibold text-emerald-700">Green = Incoming trade (bought)</span>, and{" "}
          <span className="font-semibold text-rose-700">Red = Outgoing trade (sold)</span>.
        </span>
      }
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {/* View mode toggle */}
          <div className="inline-flex rounded-lg border border-slate-300 bg-slate-100 p-0.5 text-xs font-semibold">
            <button
              type="button"
              onClick={() => setViewMode("ring")}
              className={`rounded-md px-2.5 py-1 transition-colors ${
                viewMode === "ring" ? "bg-white text-slate-900 shadow-xs" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Network Ring
            </button>
            <button
              type="button"
              onClick={() => setViewMode("hub")}
              className={`rounded-md px-2.5 py-1 transition-colors ${
                viewMode === "hub" ? "bg-white text-slate-900 shadow-xs" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Hub View (#{householdId})
            </button>
          </div>

          {/* Filter toggle */}
          <label className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={focusOnlySelected}
              onChange={(e) => setFocusOnlySelected(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
            />
            Only House #{householdId} Trades
          </label>
        </div>
      }
    >
      {/* Metrics Banner */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* Incoming (Bought) */}
        <div className="rounded-lg border border-emerald-200 bg-emerald-50/60 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-emerald-800">
            <span>Incoming (Bought)</span>
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-xl font-bold text-emerald-900">
              +{fmtNum(stats.selectedIncomingKwh)}
            </span>
            <span className="text-xs text-emerald-700">kWh</span>
          </div>
          <div className="text-[11px] text-emerald-700">
            House #{householdId} · {stats.selectedIncomingCount} trade(s) ({fmtRs(stats.selectedIncomingCost)})
          </div>
        </div>

        {/* Outgoing (Sold) */}
        <div className="rounded-lg border border-rose-200 bg-rose-50/60 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-rose-800">
            <span>Outgoing (Sold)</span>
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-rose-500" />
          </div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-xl font-bold text-rose-900">
              -{fmtNum(stats.selectedOutgoingKwh)}
            </span>
            <span className="text-xs text-rose-700">kWh</span>
          </div>
          <div className="text-[11px] text-rose-700">
            House #{householdId} · {stats.selectedOutgoingCount} trade(s) (+{fmtRs(stats.selectedOutgoingCost)})
          </div>
        </div>

        {/* Net Trading Balance */}
        <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-blue-800">
            <span>House #{householdId} Net Traded</span>
            <span className="text-xs font-bold text-blue-700">
              {stats.netKwh >= 0 ? "Surplus Seller" : "Net Buyer"}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className={`text-xl font-bold ${stats.netKwh >= 0 ? "text-emerald-800" : "text-rose-800"}`}>
              {stats.netKwh >= 0 ? "+" : ""}
              {fmtNum(stats.netKwh)}
            </span>
            <span className="text-xs text-slate-600">kWh</span>
          </div>
          <div className="text-[11px] text-slate-600">
            Net Value: {stats.netCost >= 0 ? `+${fmtRs(stats.netCost)}` : fmtRs(stats.netCost)}
          </div>
        </div>

        {/* Microgrid Total */}
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
            <span>Microgrid Total</span>
            <span className="text-[11px] text-slate-500">{edges.length} active pair(s)</span>
          </div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-xl font-bold text-slate-900">{fmtNum(stats.totalKwh)}</span>
            <span className="text-xs text-slate-600">kWh</span>
          </div>
          <div className="text-[11px] text-slate-600">
            Total Value: {fmtRs(stats.totalCost)}
          </div>
        </div>
      </div>

      {/* Time scrubbing bar */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-semibold shadow-xs transition-colors ${
              isPlaying ? "bg-amber-600 text-white hover:bg-amber-700" : "bg-blue-700 text-white hover:bg-blue-800"
            }`}
          >
            {isPlaying ? (
              <>
                <span>⏸</span> Pause
              </>
            ) : (
              <>
                <span>▶</span> Play Day
              </>
            )}
          </button>

          <button
            type="button"
            onClick={() => setSelectedHour("all")}
            className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
              selectedHour === "all"
                ? "bg-slate-800 text-white"
                : "border border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
            }`}
          >
            All 24 Hours
          </button>
        </div>

        {/* Hour buttons list */}
        <div className="flex flex-wrap items-center gap-1">
          {hours.map((h) => {
            const label = `${String(h).padStart(2, "0")}:00`;
            const isSel = selectedHour === h;
            return (
              <button
                key={h}
                type="button"
                onClick={() => {
                  setSelectedHour(h);
                  setIsPlaying(false);
                }}
                className={`rounded px-1.5 py-0.5 text-[11px] font-mono transition-colors ${
                  isSel
                    ? "bg-blue-700 font-bold text-white shadow-xs"
                    : "text-slate-700 hover:bg-slate-200 hover:text-slate-900"
                }`}
                title={`Inspect trades at ${label}`}
              >
                {String(h).padStart(2, "0")}
              </button>
            );
          })}
        </div>

        <div className="text-xs font-semibold text-slate-700">
          Viewing:{" "}
          <span className="font-bold text-blue-900">
            {selectedHour === "all" ? "Whole Day (Aggregate)" : `Slot ${String(selectedHour).padStart(2, "0")}:00`}
          </span>
        </div>
      </div>

      {/* Main SVG Visualization Container (Clean Light Theme) */}
      <div className="relative overflow-hidden rounded-xl border border-slate-300 bg-slate-50/70 p-2 shadow-xs">
        {/* Floating Legend */}
        <div className="absolute top-3 left-3 z-10 flex flex-col gap-1.5 rounded-lg border border-slate-200 bg-white/95 p-2.5 text-xs text-slate-800 shadow-md backdrop-blur-xs">
          <div className="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-9.5z"
                fill="#dbeafe"
                stroke="#2563eb"
                strokeWidth="1.5"
              />
              <path d="M9 21v-6a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v6" fill="#1e40af" />
            </svg>
            <span className="font-semibold text-slate-900">House Node</span>
            <span className="text-[10px] text-slate-500">(Click to select)</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-5 rounded-full bg-emerald-600 shadow-xs" />
            <span className="font-semibold text-emerald-800">Incoming Trade (Bought)</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-5 rounded-full bg-rose-600 shadow-xs" />
            <span className="font-semibold text-rose-800">Outgoing Trade (Sold)</span>
          </div>

          <div className="mt-0.5 border-t border-slate-200 pt-1 text-[11px] text-slate-600">
            Selected: <span className="font-bold text-blue-700">House #{householdId}</span>
          </div>
        </div>

        {/* Hover details badge */}
        {hoveredEdge && (
          <div className="absolute top-3 right-3 z-10 rounded-lg border border-slate-200 bg-white/95 p-3 text-xs text-slate-800 shadow-md backdrop-blur-xs">
            <div className="font-bold text-slate-900">
              Trade: House #{hoveredEdge.sellerId} <span className="text-rose-600 font-bold">➔</span> House #{hoveredEdge.buyerId}
            </div>
            <div className="mt-1 space-y-0.5 text-[11px] text-slate-700">
              <div>
                Volume: <span className="font-bold text-slate-900">{fmtKwh(hoveredEdge.totalKwh)} kWh</span>
              </div>
              <div>
                Cost: <span className="font-bold text-emerald-700">{fmtRs(hoveredEdge.totalCost)}</span>
              </div>
              <div>
                Avg Price:{" "}
                <span className="font-semibold text-slate-800">
                  Rs. {fmtNum(hoveredEdge.totalCost / (hoveredEdge.totalKwh || 1))}/kWh
                </span>
              </div>
              <div>Trades count: {hoveredEdge.tradeCount}</div>
            </div>
          </div>
        )}

        {/* SVG Drawing Canvas */}
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto select-none"
          style={{ minHeight: "500px", maxHeight: "640px" }}
        >
          <defs>
            {/* Arrowheads for different states */}
            <marker id="arrow-green" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
            </marker>

            <marker id="arrow-red" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#e11d48" />
            </marker>

            <marker id="arrow-neutral" viewBox="0 0 10 10" refX="19" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Background Microgrid Ring Guidelines */}
          {viewMode === "ring" && (
            <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#e2e8f0" strokeWidth="1.5" strokeDasharray="4 6" />
          )}

          {/* Trade Edges Layer */}
          <g className="edges-layer">
            {edges.map((edge) => {
              const isSelectedBuyer = edge.buyerId === activeFocus;
              const isSelectedSeller = edge.sellerId === activeFocus;
              const isHovered = hoveredEdge?.id === edge.id;

              if (focusOnlySelected && !isSelectedBuyer && !isSelectedSeller) {
                return null;
              }

              // Determine node positions based on view mode
              const p1 = viewMode === "ring" ? nodePositions.get(edge.sellerId) : hubPositions.get(edge.sellerId);
              const p2 = viewMode === "ring" ? nodePositions.get(edge.buyerId) : hubPositions.get(edge.buyerId);

              if (!p1 || !p2) return null;

              // Quadratic bezier curve bending towards center
              const midX = (p1.x + p2.x) / 2;
              const midY = (p1.y + p2.y) / 2;
              const bendFactor = viewMode === "ring" ? 0.35 : 0.15;
              const ctrlX = midX * (1 - bendFactor) + cx * bendFactor;
              const ctrlY = midY * (1 - bendFactor) + cy * bendFactor;

              // Color logic:
              // - Incoming to active house -> GREEN
              // - Outgoing from active house -> RED
              // - Otherwise subtle neutral slate
              let strokeColor = "#cbd5e1";
              let strokeWidth = Math.max(1.2, Math.min(5, (edge.totalKwh / maxKwh) * 4.5));
              let markerEnd = "url(#arrow-neutral)";
              let opacity = 0.35;

              if (isSelectedBuyer) {
                strokeColor = "#059669"; // Emerald green for incoming energy
                strokeWidth = Math.max(2.5, Math.min(6, (edge.totalKwh / maxKwh) * 5.5));
                markerEnd = "url(#arrow-green)";
                opacity = 0.95;
              } else if (isSelectedSeller) {
                strokeColor = "#e11d48"; // Rose red for outgoing energy
                strokeWidth = Math.max(2.5, Math.min(6, (edge.totalKwh / maxKwh) * 5.5));
                markerEnd = "url(#arrow-red)";
                opacity = 0.95;
              }

              if (isHovered) {
                strokeWidth += 2;
                opacity = 1;
              }

              return (
                <g key={edge.id} className="cursor-pointer" onMouseEnter={() => setHoveredEdge(edge)} onMouseLeave={() => setHoveredEdge(null)}>
                  {/* Fixed wider transparent hit area for smooth hover */}
                  <path
                    d={`M ${p1.x} ${p1.y} Q ${ctrlX} ${ctrlY} ${p2.x} ${p2.y}`}
                    fill="none"
                    stroke="transparent"
                    strokeWidth="14"
                  />
                  {/* Visible trade edge */}
                  <path
                    d={`M ${p1.x} ${p1.y} Q ${ctrlX} ${ctrlY} ${p2.x} ${p2.y}`}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeOpacity={opacity}
                    markerEnd={markerEnd}
                  />
                </g>
              );
            })}
          </g>

          {/* Household Nodes Layer (Zero Hover Jitter) */}
          <g className="nodes-layer">
            {householdList.map((h) => {
              const pos = viewMode === "ring" ? nodePositions.get(h.householdId) : hubPositions.get(h.householdId);
              if (!pos) return null;

              const isSelected = h.householdId === householdId;
              const isHovered = h.householdId === hoveredNode;

              // Check if node is actively trading with focus
              const hasIncoming = edges.some((e) => e.buyerId === h.householdId && e.sellerId === activeFocus);
              const hasOutgoing = edges.some((e) => e.sellerId === h.householdId && e.buyerId === activeFocus);

              // Light theme color palette
              let nodeBg = "#ffffff";
              let nodeStroke = "#cbd5e1";
              let iconFill = "#64748b";

              if (isSelected) {
                nodeBg = "#eff6ff";
                nodeStroke = "#2563eb";
                iconFill = "#1d4ed8";
              } else if (isHovered) {
                nodeBg = "#f8fafc";
                nodeStroke = "#475569";
                iconFill = "#334155";
              } else if (hasIncoming) {
                nodeBg = "#f0fdf4";
                nodeStroke = "#10b981";
                iconFill = "#059669";
              } else if (hasOutgoing) {
                nodeBg = "#fff1f2";
                nodeStroke = "#f43f5e";
                iconFill = "#e11d48";
              }

              const nodeRadius = 18;

              return (
                <g
                  key={h.householdId}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer"
                  onClick={() => onSelectHousehold(h.householdId)}
                  onMouseEnter={() => setHoveredNode(h.householdId)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  {/* Fixed invisible hit circle: stops any cursor flicker / jitter completely */}
                  <circle r={28} fill="transparent" />

                  {/* Outer ring for selected house */}
                  {isSelected && (
                    <circle
                      r={nodeRadius + 4}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2"
                      strokeDasharray="4 3"
                    />
                  )}

                  {/* Circular Node base */}
                  <circle
                    r={nodeRadius}
                    fill={nodeBg}
                    stroke={nodeStroke}
                    strokeWidth={isSelected ? 2.5 : isHovered ? 2 : 1.5}
                  />

                  {/* House Symbol Icon (clean static geometry) */}
                  <g transform="translate(-8, -9) scale(0.68)">
                    {/* House roof */}
                    <path
                      d="M2 10.5 L12 2.5 L22 10.5 L20 10.5 L20 20 A 1 1 0 0 1 19 21 L5 21 A 1 1 0 0 1 4 20 L4 10.5 Z"
                      fill={iconFill}
                      stroke={iconFill}
                      strokeWidth="1.2"
                      strokeLinejoin="round"
                    />
                    {/* House door */}
                    <path d="M9 21 L9 13.5 A 1 1 0 0 1 10 12.5 L14 12.5 A 1 1 0 0 1 15 13.5 L15 21 Z" fill={nodeBg} />
                    {/* Chimney */}
                    <path d="M17 4 L19 4 L19 8 L17 6.4 Z" fill={iconFill} />
                  </g>

                  {/* Household ID Tag */}
                  <text
                    y={nodeRadius + 14}
                    textAnchor="middle"
                    fill={isSelected ? "#1e40af" : "#334155"}
                    fontSize={isSelected ? "12" : "11"}
                    fontWeight={isSelected ? "bold" : "600"}
                    fontFamily="monospace"
                  >
                    #{h.householdId}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Empty state notice if no trades */}
        {edges.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 text-center backdrop-blur-xs">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" className="mb-2">
              <path d="M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-9.5z" />
              <path d="M9 21v-6a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v6" />
            </svg>
            <p className="text-sm font-semibold text-slate-700">No trades executed for this slot/date</p>
            <p className="text-xs text-slate-500">Run the simulation to generate peer-to-peer microgrid trades.</p>
          </div>
        )}
      </div>

      {/* Caption note */}
      <div className="mt-3 flex flex-wrap items-center justify-between text-xs text-slate-600">
        <div>
          Tip: Click any house node to select it and update the entire dashboard. Hover over any trade line to inspect
          volume and price.
        </div>
        <div className="font-mono text-[11px] font-semibold text-slate-700">
          {edges.length} active trade routes · {filteredTrades.length} transactions
        </div>
      </div>
    </Section>
  );
}
