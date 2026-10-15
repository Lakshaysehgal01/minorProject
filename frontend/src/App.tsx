import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { EpochProvider } from "./hooks/EpochContext";
import { useHealth, useHouseholds, useStats } from "./hooks/resources";
import { useSimulation } from "./hooks/useSimulation";
import { DashboardPage } from "./pages/DashboardPage";
import { ModelResultsPage } from "./pages/ModelResultsPage";
import { lastSimulatedDate, REPLAY_MAX, REPLAY_MIN } from "./utils/date";
import { USE_MOCK } from "./api/client";


type Tab = "dashboard" | "results";


const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "results", label: "Model Results" },
];


function Shell() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [date, setDate] = useState<string | null>(null);
  const [householdId, setHouseholdId] = useState(0);


  const stats = useStats();
  const health = useHealth();
  const households = useHouseholds();
  const sim = useSimulation((ranDate) => setDate(ranDate));


  // Start on the most recently simulated day (or the first replay day if nothing ran yet).
  useEffect(() => {
    if (date !== null) return;
    if (stats.data) setDate(lastSimulatedDate(stats.data.simulation.cursor, stats.data.simulation.slotsCompleted));
    else if (stats.error) setDate(REPLAY_MIN);
  }, [date, stats.data, stats.error]);


  function changeDate(next: string) {
    setDate(next < REPLAY_MIN ? REPLAY_MIN : next > REPLAY_MAX ? REPLAY_MAX : next);
  }


  return (
    <div className="min-h-screen">
      {USE_MOCK && (
        <div className="bg-amber-200 px-5 py-1 text-center text-base font-bold text-amber-950">
          MOCK MODE: showing generated demo data, not the real backend
        </div>
      )}
      <Header
        date={date}
        onDateChange={changeDate}
        stats={stats}
        health={health}
        running={sim.running}
        elapsed={sim.elapsed}
        runError={sim.error}
        onDismissError={sim.dismissError}
        onRunDay={sim.runDay}
        onRunTick={sim.runTick}
      />


      <nav className="border-b border-slate-300 bg-white" aria-label="Pages">
        <div className="mx-auto flex max-w-[1500px] gap-2 px-5">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
              className={`-mb-px border-b-4 px-5 py-3 text-lg font-bold ${
                tab === t.id ? "border-blue-700 text-blue-900" : "border-transparent text-slate-700 hover:text-slate-900"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>


      <main className="mx-auto max-w-[1500px] px-5 py-6">
        {tab === "dashboard" ? (
          <DashboardPage date={date} householdId={householdId} onHouseholdChange={setHouseholdId} households={households} />
        ) : (
          <ModelResultsPage stats={stats} />
        )}
      </main>
    </div>
  );
}


export default function App() {
  return (
    <EpochProvider>
      <Shell />
    </EpochProvider>
  );
}



