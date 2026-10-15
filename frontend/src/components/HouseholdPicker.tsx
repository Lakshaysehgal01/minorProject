import type { Resource } from "../hooks/useResource";
import type { Household, Tier } from "../types/api";
import { fmtSigned } from "../utils/format";
import { AsyncState } from "./AsyncState";
import { Badge, type Tone } from "./Badge";
import { Section } from "./Section";


export const TIER_TONE: Record<Tier, Tone> = { small: "slate", medium: "blue", large: "amber" };


export function EnergyBalance({ kwh }: { kwh: number }) {
  const tone = kwh > 0 ? "text-emerald-800" : kwh < 0 ? "text-red-800" : "text-slate-700";
  const word = kwh > 0 ? "net buyer" : kwh < 0 ? "net seller" : "even";
  return (
    <span className={`font-bold tabular-nums ${tone}`} title={`${word}: energy received minus energy sold`}>
      {fmtSigned(kwh)} kWh
    </span>
  );
}


export function HouseholdPicker({
  households,
  selected,
  onSelect,
}: {
  households: Resource<Household[]>;
  selected: number;
  onSelect: (id: number) => void;
}) {
  return (
    <Section
      title="Households"
      subtitle="Select a household to see its charts, forecast and trades. Balance = on-chain energy received minus sold (+ net buyer, − net seller)."
    >
      <AsyncState resource={households} isEmpty={(d) => d.length === 0} emptyMessage="No households returned.">
        {(list) => (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {list.map((h) => {
              const active = h.householdId === selected;
              return (
                <button
                  key={h.householdId}
                  type="button"
                  onClick={() => onSelect(h.householdId)}
                  aria-pressed={active}
                  className={`rounded-lg border-2 p-3 text-left transition ${
                    active ? "border-blue-700 bg-blue-50 ring-2 ring-blue-700" : "border-slate-300 bg-white hover:border-slate-500"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-lg font-extrabold">#{h.householdId}</span>
                    <Badge tone={TIER_TONE[h.tier]}>{h.tier}</Badge>
                  </div>
                  <p className="mt-1 text-base text-slate-800">{h.panelCapacityKw.toFixed(2)} kW panels</p>
                  <p className="text-base">
                    <EnergyBalance kwh={h.energyBalanceKwh} />
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </AsyncState>
    </Section>
  );
}



