import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import { runDay, runTick } from "../api/endpoints";
import { dateOf } from "../utils/date";
import { useEpoch } from "./EpochContext";


export type RunKind = "day" | "tick";


// Runs a simulation request, then invalidates all cached data so every section refetches.
// `onRan` receives the date that was just simulated so the UI can jump to it.
export function useSimulation(onRan: (date: string) => void) {
  const { invalidateAll } = useEpoch();
  const [running, setRunning] = useState<RunKind | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const onRanRef = useRef(onRan);
  onRanRef.current = onRan;


  useEffect(() => {
    if (!running) return;
    setElapsed(0);
    const started = Date.now();
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 500);
    return () => window.clearInterval(id);
  }, [running]);


  const run = useCallback(
    async (kind: RunKind) => {
      setError(null);
      setRunning(kind);
      try {
        const date = kind === "day" ? (await runDay()).date : dateOf((await runTick()).slot);
        invalidateAll();
        onRanRef.current(date);
      } catch (e) {
        setError(errorMessage(e));
        invalidateAll(); // a failed run may still have advanced part of the day
      } finally {
        setRunning(null);
      }
    },
    [invalidateAll]
  );


  return {
    running,
    elapsed,
    error,
    dismissError: () => setError(null),
    runDay: () => run("day"),
    runTick: () => run("tick"),
  };
}



