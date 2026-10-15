import type { RunKind } from "../hooks/useSimulation";
import type { Resource } from "../hooks/useResource";
import type { HealthResponse, Stats } from "../types/api";
import { REPLAY_MAX, REPLAY_MIN, slotLabel } from "../utils/date";
import { fmtInt } from "../utils/format";
import { HealthBadges } from "./HealthBadges";
import { Spinner } from "./Spinner";


interface Props {
  date: string | null;
  onDateChange: (date: string) => void;
  stats: Resource<Stats>;
  health: Resource<HealthResponse>;
  running: RunKind | null;
  elapsed: number;
  runError: string | null;
  onDismissError: () => void;
  onRunDay: () => void;
  onRunTick: () => void;
}


export function Header({ date, onDateChange, stats, health, running, elapsed, runError, onDismissError, onRunDay, onRunTick }: Props) {
  const sim = stats.data?.simulation;
  const finished = sim?.complete ?? false;
  const disabled = running !== null || finished;


  return (
    <header className="border-b-2 border-slate-300 bg-white">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 py-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900">Smart Energy Microgrid Trading</h1>
          <p className="text-sm text-slate-700">
            {sim
              ? finished
                ? `Replay finished: all ${fmtInt(sim.totalSlots)} hours simulated`
                : `Next slot: ${slotLabel(sim.cursor)} · ${fmtInt(sim.slotsCompleted)} / ${fmtInt(sim.totalSlots)} hours simulated`
              : "Peer-to-peer solar trading on a local blockchain"}
          </p>
        </div>


        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-base font-semibold text-slate-900">
            Date
            <input
              type="date"
              value={date ?? ""}
              min={REPLAY_MIN}
              max={REPLAY_MAX}
              onChange={(e) => e.target.value && onDateChange(e.target.value)}
              className="rounded-md border-2 border-slate-500 bg-white px-2 py-1 text-base font-medium"
            />
          </label>


          <button
            type="button"
            onClick={onRunDay}
            disabled={disabled}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-700 px-5 py-2.5 text-lg font-bold text-white shadow hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {running === "day" && <Spinner />}
            {running === "day" ? "Running…" : "Run Next Day"}
          </button>
          <button
            type="button"
            onClick={onRunTick}
            disabled={disabled}
            className="inline-flex items-center gap-2 rounded-lg border-2 border-blue-700 bg-white px-3 py-1.5 text-base font-semibold text-blue-800 hover:bg-blue-50 disabled:cursor-not-allowed disabled:border-slate-400 disabled:text-slate-500"
          >
            {running === "tick" && <Spinner className="h-4 w-4" />}
            Run 1 Hour
          </button>
        </div>
      </div>


      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-3 px-5 pb-3">
        <HealthBadges health={health} />
        {running && (
          <p className="text-base font-semibold text-blue-900" role="status">
            {running === "day"
              ? `Simulating 24 hourly slots: predicting, matching and settling trades on-chain (about 30 s)… ${elapsed}s elapsed`
              : `Simulating one hourly slot… ${elapsed}s elapsed`}
          </p>
        )}
      </div>


      {runError && (
        <div role="alert" className="mx-auto mb-3 flex max-w-[1500px] items-start justify-between gap-3 rounded-lg border-2 border-red-600 bg-red-50 px-4 py-3">
          <p className="text-base font-medium text-red-900">Simulation failed: {runError}</p>
          <button type="button" onClick={onDismissError} className="text-sm font-semibold text-red-900 underline">
            Dismiss
          </button>
        </div>
      )}
    </header>
  );
}



